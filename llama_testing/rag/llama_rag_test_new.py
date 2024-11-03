import logging
import sys
import torch
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core import (
    VectorStoreIndex,
    PromptTemplate,
    ServiceContext,
    StorageContext,
)
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.core import get_response_synthesizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import re

# Logging configuration
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

# Model names
LLAMA2_7B = "meta-llama/Llama-2-7b-hf"
LLAMA2_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"
LLAMA2_13B = "meta-llama/Llama-2-13b-hf"
LLAMA2_13B_CHAT = "meta-llama/Llama-2-13b-chat-hf"
LLAMA2_70B = "meta-llama/Llama-2-70b-hf"
LLAMA2_70B_CHAT = "meta-llama/Llama-2-70b-chat-hf"
LLAMA3_8B_INSTRUCT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CUSTOM_MODEL = "hyonbokan/bgp-llama-knowledge-5k"

BGE_SMALL = "BAAI/bge-small-en-v1.5" # small is better than large
BGE_LARGE = "BAAI/bge-large-en-v1.5"
BGE_ICL = "BAAI/bge-en-icl"
BGE_M3 = "BAAI/bge-m3"
llm_embed = "BAAI/llm-embedder"
STELLA = "dunzhang/stella_en_1.5B_v5"
NV_EMBED = "nvidia/NV-Embed-v2"

csv_path = "/home/hb/dataset_bgp/candidates"
json_path = "/home/hb/dataset_bgp/bgp_tab_rag_test/euro_1136/json/"
narrative = "/home/hb/dataset_bgp/bgp_tab_rag_test/euro_1136/narrative"
new_nar = "/home/hb/dataset_bgp/bgp_tab_rag_test/euro_1136/new_nar"
process_files = "/home/hb/dataset_bgp/bgp_tab_rag_test/euro_1136/process_file"
five_min_summary = "/home/hb/dataset_bgp/bgp_tab_rag_test/euro_1136/struct_text"


SYSTEM_PROMPT = """
You are an AI assistant that answers questions in a friendly manner, based on the given source BGP data. Here are some rules you always follow:
- Generate only the requested output, don't include any other language before or after the requested output.
- Your answers should be clear and include relevant information such as timestamps or corresponding values when analyzing BGP data features.
- If the prompt includes the word 'collect' related to BGP data, first provide a snapshot of the collected data, and then summarize it.
- Never say thank you, that you are happy to help, that you are an AI agent, and additional suggestions.
"""
# Prompt template
query_wrapper_prompt = PromptTemplate(
    "[INST] <<SYS>>\n" + SYSTEM_PROMPT + "\n<</SYS>>\n\n{query_str} [/INST] "
)

# Initialize models and settings
def initialize_models():
    # Initialize the LLM
    llm = HuggingFaceLLM(
        model_name=LLAMA3_8B_INSTRUCT,
        tokenizer_name=LLAMA3_8B_INSTRUCT,
        context_window=4096,
        max_new_tokens=512,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        device_map="auto",
        model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": False},
    )

    # Initialize the embedding model
    embed_model = HuggingFaceEmbedding(model_name=BGE_SMALL)

    # Create a service context with the models
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        query_wrapper_prompt=query_wrapper_prompt,
    )

    return service_context

# Function to extract metadata from document text
def extract_metadata(text):
    # Extract timestamp and AS number using regex
    timestamp = "Unknown"
    as_number = "Unknown"

    timestamp_match = re.search(r"On (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),", text)
    as_number_match = re.search(r"Autonomous System (\d+)", text)

    if timestamp_match:
        timestamp = timestamp_match.group(1)
    if as_number_match:
        as_number = as_number_match.group(1)

    return {
        "timestamp": timestamp,
        "as_number": as_number,
    }

# Load documents with metadata and custom chunking
def load_documents_with_metadata_and_chunking(
    directory_path, chunk_size=1500, chunk_overlap=200
):
    documents = []
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        # Determine document type based on filename
        if "overall_summary" in filename:
            document_type = "overall_summary"
        elif "data_point_summaries" in filename:
            document_type = "data_point_summary"
        elif "prefix_announcements" in filename:
            document_type = "prefix_announcements"
        elif "prefix_withdrawals" in filename:
            document_type = "prefix_withdrawals"
        elif "updates_per_peer" in filename:
            document_type = "updates_per_peer"
        else:
            document_type = "unknown"

        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()

            # Extract metadata from the text
            metadata = extract_metadata(text)
            # Add document_type to metadata
            metadata["document_type"] = document_type

            # Custom chunking with overlap
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk_text = text[i : i + chunk_size]
                chunk = Document(text=chunk_text, metadata=metadata)
                documents.append(chunk)

    return documents

# Function to determine document type from query
def determine_document_type_from_query(query):
    query = query.lower()
    if "overall summary" in query or "min max" in query:
        return "overall_summary"
    elif "data point" in query or "timestamp" in query:
        return "data_point_summary"
    elif "prefix announcement" in query or "prefixes announced" in query:
        return "prefix_announcements"
    elif "prefix withdrawal" in query or "prefixes withdrawn" in query:
        return "prefix_withdrawals"
    elif "updates per peer" in query or "peer updates" in query:
        return "updates_per_peer"
    else:
        return None  # No specific document type identified

# Custom scoring function
def custom_scoring_function(query_embedding, doc_embedding, doc_metadata=None):
    # Compute cosine similarity
    similarity_score = np.dot(query_embedding, doc_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
    )

    # Adjust score based on metadata (example: boost documents from AS number 1136)
    if doc_metadata and doc_metadata.get("as_number") == "1136":
        similarity_score += 0.1  # Boost score for AS 1136

    return similarity_score

# Query BGP data with filters and custom scoring
def query_bgp_data_with_filters_and_custom_scoring(
    index,
    query,
    service_context,
    document_type=None,
    similarity_top_k=3,
    similarity_cutoff=0.0,
):
    # Define metadata filters based on document_type
    if document_type:
        metadata_filters = MetadataFilters(
            filters=[MetadataFilter(key="document_type", value=document_type)]
        )
    else:
        metadata_filters = None  # No filters if document_type is not specified

    # Create a retriever with custom scoring function
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k,
        filters=metadata_filters,
        similarity_cutoff=similarity_cutoff,
        retriever_mode="default",
        embedding_fn=service_context.embed_model.get_text_embedding,
        similarity_fn=custom_scoring_function,
    )

    # Create a response synthesizer
    response_synthesizer = get_response_synthesizer(service_context=service_context)

    # Create a query engine with the retriever and response synthesizer
    query_engine = RetrieverQueryEngine(
        retriever=retriever, response_synthesizer=response_synthesizer
    )

    # Start query and get the response
    response = query_engine.query(query)

    # Output retrieved documents
    print("\nRetrieved Documents:")
    print("-" * 80)
    for idx, node in enumerate(response.source_nodes):
        print(f"\nDocument {idx + 1}:")
        print(f"Score: {node.score}")
        print(f"Metadata: {node.node.metadata}")
        print(f"Content:\n{node.node.get_content()}")

    # Output the response
    print("-" * 80)
    print("\nResponse:")
    print(response.response)

# Main function with continuous querying
def main():
    logger.info("Initializing models...")
    service_context = initialize_models()

    logger.info("Loading documents with metadata and custom chunking...")
    documents = load_documents_with_metadata_and_chunking(
        five_min_summary, chunk_size=1500, chunk_overlap=200
    )

    logger.info("Creating vector store index...")
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        service_context=service_context,
    )

    # Continuous querying loop
    while True:
        query = input("Enter your query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting the program.")
            break

        logger.info(f"Running query: {query}")

        # Determine the document type from the query
        document_type = determine_document_type_from_query(query)

        # Adjust similarity_top_k as needed
        similarity_top_k = 3
        similarity_cutoff = 0.1  # Adjust as needed

        query_bgp_data_with_filters_and_custom_scoring(
            index,
            query,
            service_context=service_context,
            document_type=document_type,
            similarity_top_k=similarity_top_k,
            similarity_cutoff=similarity_cutoff,
        )

if __name__ == "__main__":
    main()