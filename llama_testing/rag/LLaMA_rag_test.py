import logging
import sys
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

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

# System prompt
SYSTEM_PROMPT = """
You are an AI assistant that answers questions in a friendly manner, based on the given source BGP data. Here are some rules you always follow:
- Generate only the requested output, don't include any other language before or after the requested output.
- Your answers should be direct and include relevant timestamps when analyzing BGP data features.
- Check the collected BGP data given below. Each row represents the features collected over a specific period.
- Never say thank you, that you are happy to help, that you are an AI agent, and additional suggestions.
"""

# Prompt template
query_wrapper_prompt = PromptTemplate("[INST]<>\n" + SYSTEM_PROMPT + "<>\n\n{query_str}[/INST] ")

# Initialize models and settings
def initialize_models():
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=512,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name=LLAMA3_8B_INSTRUCT,
        model_name=LLAMA3_8B_INSTRUCT,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": False},
    )

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Set the models to the global Settings
    Settings.llm = llm
    Settings.embed_model = embed_model

    return llm, embed_model

# Load documents from directory
def load_documents(directory_path):
    reader = SimpleDirectoryReader(directory_path)
    documents = reader.load_data()
    return documents

# Create vector store index
def create_index(documents):
    index = VectorStoreIndex.from_documents(documents)
    return index

# Query the engine with streaming support
def query_bgp_data(index, query):
    query_engine = index.as_query_engine(streaming=True)
    
    # Start query and get the streaming response generator
    response = query_engine.query(query)
    
    # Stream tokens to the console
    for token in response.response_gen:
        print(token, end="")
        sys.stdout.flush()  # Flush to ensure immediate output

# Main function
def main():
    logger.info("Initializing models...")
    llm, embed_model = initialize_models()

    logger.info("Loading documents...")
    documents = load_documents("/home/hb/dataset_bgp/bgp_tab_rag_test")

    logger.info("Creating vector store index...")
    index = create_index(documents)

    query = "What is BGP in computer network?"
    logger.info(f"Running query: {query}")
    query_bgp_data(index, query)

if __name__ == "__main__":
    main()