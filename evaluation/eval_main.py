import transformers
from transformers import pipeline
import torch
from torch import cuda
import os

def main():
    # User input for model ID
    model_id = input("Enter the model ID: ")

    # Check for CUDA availability
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    # Assuming the user has an Hugging Face token set in their environment
    hf_auth = os.environ.get('hf_token')

    # Load model configuration
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    # Load the model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        device_map='auto',
        use_auth_token=hf_auth
    )

    model.eval()
    print(f"Model loaded on {device}")

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Set logging verbosity to critical
    transformers.logging.set_verbosity(transformers.logging.CRITICAL)

    # Create pipeline
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=712)

    # Loop for continuous prompting
    while True:
        user_prompt = input("\nEnter your prompt (or type 'exit' to quit): ")
        if user_prompt.lower() == 'exit':
            break

        # Generate text
        result = pipe(f"<s>[INST] {user_prompt} [/INST]")
        print(result[0]['generated_text'])

if __name__ == "__main__":
    main()