import os
import json
import logging
from threading import Lock, Thread
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig, 
    TextIteratorStreamer
)

# Initialize model, tokenizer, and streamer variables
model = None
tokenizer = None
streamer = None
model_lock = Lock()

def load_model():
    global model, tokenizer, streamer
    with model_lock:
        if model is None or tokenizer is None or streamer is None:
            try:
                model_id = 'meta-llama/Llama-2-7b-chat-hf'
                hf_auth = os.environ.get('hf_token')

                model_config = AutoConfig.from_pretrained(
                    model_id,
                    use_auth_token=hf_auth
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    config=model_config,
                    device_map='auto',
                    use_auth_token=hf_auth
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    use_auth_token=hf_auth
                )

                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.padding_side = "right"

                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                logging.info("Model loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load the model: {str(e)}")
                raise

    return model, tokenizer, streamer

def generate_response(prompt):
    model, tokenizer, streamer = load_model()
    
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    generation_kwargs = dict(
        inputs=inputs["input_ids"],  # Pass the tensor directly
        streamer=streamer,
        max_new_tokens=756,  # Adjust based on your requirements
        # do_sample=True,
        # top_p=1.0,
        # temperature=float(0.6),  # Adjust for creativity vs. accuracy
        # top_k=1,  # Use only the top K tokens for generation
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    generated_text = ""
    
    # Collect the output from the streamer
    for new_text in streamer:
        generated_text += new_text
        print(new_text, end="")  # Output in real-time in Jupyter Notebook
    
    thread.join()  # Ensure the thread has finished before returning the result
    
    return generated_text.strip()

# Example usage in a Jupyter Notebook
prompt = """
You are an AI assistant and your task is to answers the user query on the given BGP data. Here are some rules you always follow:
- Generate only the requested output, don't include any other language before or after the requested output.
- Your answers should be direct and without any suggestions. 
- Check the collected BGP data given below. Each row represents the features collected over a specific period. identify and extract specific details like number of announcements, timestamps, and other relevant features.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
Answer this user query: Collect real-time BGP data for AS3356. Provide summary on the BGP data.Here is the summary of the collected BGP data:
[TLE] The context is about BGP update message for analysis. The section is related to a specific time period of BGP monitoring. [TAB] col: | Timestamp | Autonomous System Number | Number of Routes | Number of New Routes | Number of Withdrawals | Number of Origin Changes | Number of Route Changes | Maximum Path Length | Average Path Length | Maximum Edit Distance | Average Edit Distance | Number of Announcements | Total Withdrawals | Number of Unique Prefixes Announced |row 1: | 2024-08-27 16:13:52 | 3356 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6722 | 0 | 0 | [SEP]
"""
response = generate_response(prompt)
print("\nFinal generated response:")
print(response)
