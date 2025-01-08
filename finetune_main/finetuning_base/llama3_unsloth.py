import torch
from unsloth import FastLanguageModel
import torch
from torch import cuda, bfloat16
from datasets import load_dataset
import os

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Available device: {DEVICE}")

model_id = "unsloth/Meta-Llama-3.1-8B",

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

max_seq_length = 2048
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = "meta-llama/Meta-Llama-3.1-8B-Instruct",
)

model.eval()
print(f"Model loaded on {device}")


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

data = load_dataset("json", data_files="/home/hb/LLM-research/dataset/BGP/PyBGPStream/PyBGPStream_main10K.json")
data["train"]

CUTOFF_LEN = 2048

def generate_prompt(data_point):
    """
    Create the text prompt from your instruction, input, and output fields.
    """
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""

def tokenize(prompt, add_eos_token=True):
    """
    Tokenizes the prompt. Optionally pads to max_length=2048 and appends an EOS token.
    Copies input_ids to labels for causal LM.
    """
    # Here, we use padding="max_length" to get uniform-length sequences of 2048.
    # Alternatively, you can use padding=False and rely on a data collator.
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",   # or padding=False + data_collator
        return_tensors=None,    # return raw Python lists
    )

    input_ids = result["input_ids"]
    attention_mask = result["attention_mask"]

    # Optionally place an EOS token at the very end if there's room
    if (
        add_eos_token
        and len(input_ids) == CUTOFF_LEN
        and input_ids[-1] != tokenizer.eos_token_id
    ):
        # Replace last token with EOS if you'd like
        input_ids[-1] = tokenizer.eos_token_id
        attention_mask[-1] = 1

    labels = input_ids.copy()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def generate_and_tokenize_prompt(data_point):
    """
    Combines prompt generation with tokenization.
    """
    full_prompt = generate_prompt(data_point)
    return tokenize(full_prompt)

# Example: split the "train" set into train/val
train_val = data["train"].train_test_split(test_size=1000, shuffle=True, seed=42)
train_data = train_val["train"].map(generate_and_tokenize_prompt)
val_data   = train_val["test"].map(generate_and_tokenize_prompt)


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_data,
    eval_dataset = val_data,
    dataset_text_field = "output",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 5000,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

trainer_stats = trainer.train()

new_model = "/home/hb/dataset_bgp/finetuned_models/LLaMA3-8B-analysis-5k-unsloth"

trainer.model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)