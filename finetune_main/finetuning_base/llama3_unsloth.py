import os
import torch
from torch import cuda, bfloat16
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

# Configuration constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_ID = "unsloth/Meta-Llama-3.1-8B"
MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct"
CUTOFF_LEN = 2048
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
DTYPE = None  # Auto-detection; use bfloat16 for Ampere GPUs
OUTPUT_DIR = "outputs"
FINETUNED_MODEL_PATH = "/home/hb/dataset_bgp/finetuned_models/LLaMA3-8B-analysis-5k-unsloth"


def setup_device():
    """Configures the device for PyTorch."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available device: {DEVICE}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    return device


def load_model():
    """Loads the pre-trained model and tokenizer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        # token="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    model.eval()
    print(f"Model loaded on {DEVICE}")
    return model, tokenizer


def configure_model_for_training(model):
    """Configures the model for LoRA-based training."""
    return FastLanguageModel.get_peft_model(
        model,
        r=16,  # Suggested: 8, 16, 32, etc.
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )


def generate_prompt(data_point):
    """Generates a text prompt from a dataset entry."""
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""


def tokenize(prompt, tokenizer, add_eos_token=True):
    """Tokenizes the prompt and optionally adds an EOS token."""
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
        return_tensors=None,
    )

    input_ids = result["input_ids"]
    attention_mask = result["attention_mask"]

    if add_eos_token and len(input_ids) == CUTOFF_LEN and input_ids[-1] != tokenizer.eos_token_id:
        input_ids[-1] = tokenizer.eos_token_id
        attention_mask[-1] = 1

    labels = input_ids.copy()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def generate_and_tokenize_prompt(data_point, tokenizer):
    """Combines prompt generation and tokenization."""
    full_prompt = generate_prompt(data_point)
    return tokenize(full_prompt, tokenizer)


def prepare_data(tokenizer):
    """Loads and processes the dataset."""
    data = load_dataset("json", data_files="/home/hb/LLM-research/openai/generated_instructions/test_3.json")
    train_val = data["train"].train_test_split(test_size=200, shuffle=True, seed=42)

    train_data = train_val["train"].map(lambda x: generate_and_tokenize_prompt(x, tokenizer))
    val_data = train_val["test"].map(lambda x: generate_and_tokenize_prompt(x, tokenizer))

    return train_data, val_data


def train_model(model, tokenizer, train_data, val_data):
    """Trains the model using the SFTTrainer."""
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field="output",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            warmup_steps=5,
            max_steps=5000,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=OUTPUT_DIR,
            report_to="none",
        ),
    )

    trainer_stats = trainer.train()
    return trainer


def save_model(trainer):
    """Saves the trained model and tokenizer."""
    trainer.model.save_pretrained(FINETUNED_MODEL_PATH)
    trainer.tokenizer.save_pretrained(FINETUNED_MODEL_PATH)
    print(f"Model saved to {FINETUNED_MODEL_PATH}")


def main():
    device = setup_device()
    model, tokenizer = load_model()
    model = configure_model_for_training(model)
    train_data, val_data = prepare_data(tokenizer)
    trainer = train_model(model, tokenizer, train_data, val_data)
    save_model(trainer)


if __name__ == "__main__":
    main()