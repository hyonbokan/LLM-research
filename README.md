# LLaMA fine-tuning

## 1. Self-instruct Framework
The Self-instruct Framework relies on two main components for generating effective training data: prompts and manual seed tasks. 

For a detailed understanding and background of the Self-instruct Framework, refer to the paper available in the repository: [Self-Instruct Framework Paper](/reference/self-instruct.pdf).

### Prompt and Manual Seed Tasks

- **Prompts**: These serve as guiding directives for the model, ensuring that the generated outputs are not only relevant but also diverse and in alignment with specific criteria. Prompts play a crucial role in instructing and controlling Chat-GPT during the instruction generation process. They set clear parameters and expectations, dictating the nature and scope of the model's output in response to varied scenarios or tasks. 

For practical examples of how prompts are utilized in specific contexts, explore different prompts used for BGP-LLaMA. These include prompts for [general BGP knowledge](/dataset/BGP/prompt_knowledge.txt), [use of PyBGPStream library](/dataset/BGP/prompt_pybgpstream.txt), and [PyBGPStream real-time analysis](/dataset/BGP/prompt_pybgpstream_realtime.txt).

- **Manual Seed Tasks**: These are manually crafted tasks that supply the model with examples of the desired output. They are crucial for instructing the model on how to appropriately respond to the prompts. Just like prompts, seed tasks are created with a particular function in mind. In the case of BGP-LLaMA, you can find seed tasks made for different BGP knowledge areas and functionalities in [dataset](/dataset/BGP/).

### Running the Self-instruct Code

- **utils.py**: The script manages the connection to OpenAI's GPT models and handles the communication process. The script manages decoding arguments, rate-limiting with sleep intervals, batching of prompts for efficiency, and the handling of API responses. 
`IMPORTANT!`: After the OpenAI API updates in 2024, older models are deprecated. Thus, `utils.py` must be updated to align with the current API standards and model availability.

- **generate_instruction.py**: The main script for the self-instruct framework. Before running it, you need to modify the script to specify the directory paths for the [prompts](/images/self_instruct1.png) and [seed tasks](/images/self_instruct2.png). The script then loads, encodes these components, and generates new sets of instructions. 
To execute the script, use the following command line example, ensuring to replace `--model_name` with the current model you intend to use and adjust `--num_instructions_to_generate` as needed:

```shell
python -m generate_instruction generate_instruction_following_data --num_instructions_to_generate=1000 --model_name="text-davinci-003"
```

## 2. Instruction Fine-tuning
**Prerequisites**:

1. **Hugging Face Account**: To access model weights and tokenizers from Hugging Face, you need to [register for an account](https://huggingface.co/join) on Hugging Face. Once registered, generate a personal access token (PAT) from your account settings under the [Access Tokens section](https://huggingface.co/settings/tokens).

2. **Environment Setup**: Ensure Python and the necessary libraries (`transformers`, `torch`, etc.) are installed in your environment. Use a virtual environment for a cleaner setup.


### Installing Dependencies

Start by installing all necessary dependencies listed in `requirements.txt` to ensure your environment is properly set up for the fine-tuning process.

```shell
pip install -r requirements.txt
```

### Fine-tuning Script
The primary script for instruction fine-tuning of LLaMA model for specific tasks such as 5G data analysis and BGP routing analysis is located in this [directory](/LLM-research/finetune_main). 

#### Model & Tokenizer Loading
```python
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map="auto",
    use_auth_token=hf_auth
)
```
- The script starts by specifying a `model_id` that identifies which model to load from Hugging Face. For fine-tuning tasks mentioned, we use `meta-llama/Llama-2-13b-chat-hf`.
- Quantization is enabled via `BitsAndBytesConfig` to reduce GPU memory usage, crucial for accommodating large models. More details on this technique can be found in the [research paper](https://arxiv.org/pdf/2312.12148.pdf).
- Authentication with Hugging Face is required to access private or restricted models and features. Set your Hugging Face authentication token as an environment variable (`hf_token`).

```python
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
```
The tokenizer is essential for converting input text into a format that the model can understand, and vice versa. It ensures that the text input is appropriately preprocessed (padding, tokenization) for the model. Setting tokenizer.pad_token = tokenizer.eos_token ensures that padding is handled correctly by using the end-of-sequence token as the padding token.

#### Data Loading and Processing
In this part, we load training data from a specified JSON file, indicating the dataset's location. All training data for different fine-tuning tasks is organized under the [finetune_main](/LLM-research/dataset) directory for various purposes.

```python
data = load_dataset("json", data_files="/home/hb/LLM-research/dataset/5G/Mobile_LLaMA_1.json")

train_val = data["train"].train_test_split(
    test_size=1300, shuffle=True, seed=42
)
train_data = (
    train_val["train"].map(generate_and_tokenize_prompt)
)
val_data = (
    train_val["test"].map(generate_and_tokenize_prompt)
)

```
The dataset undergoes processing to format it into a structure that is conducive to model training. This involves generating prompts from the data points, which are then tokenized. The process is as follows:

- Prompt Generation: The function `generate_prompt` combines instructions and outputs into a format that mirrors the input the model expects during fine-tuning.
- Tokenization: The function `tokenize` Converts the generated prompts into a sequence of tokens, making it understandable and usable by the model. This includes truncation and the addition of end-of-sequence tokens as necessary.
- Applying Processing: The processed data is applied to both the training and validation datasets. This ensures that all data fed into the model during the fine-tuning process is in the correct format, allowing for efficient and effective model training.

#### Finetuning - Low-Rank Adaptation (LoRA)
LoRA is a method designed to update only a small portion of the model parameters during the fine-tuning process. This approach significantly reduces the computational cost and memory footprint, making it feasible to fine-tune large models on limited resources. 

The key parameters in LoRA configuration:
- **lora_alpha**: Controls the scaling of the LoRA parameters. A value of 16 is a good starting point, balancing between the model's adaptability and its original knowledge retention.
- **lora_dropout**: Specifies the dropout rate for the LoRA layers, set at 0.1 to prevent overfitting.
- **lora_r**: Defines the rank of the adaptation, set at 64, indicating the size of the low-rank matrices.
- **bias**: Set to "none" to indicate that no bias is used in the LoRA adaptation.
- **task_type**: Specifies the type of task, in this case, "CAUSAL_LM" for causal language modeling.

#### Training Configuration
- **output_dir**: The directory where the output files will be saved.
- **per_device_train_batch_size**: Batch size per device, set to 4 to balance between training speed and memory usage.
- **gradient_accumulation_steps**: Number of steps to accumulate gradients before updating model parameters.
- **optim**: Specifies the optimizer, using "paged_adamw_32bit" for efficient memory usage.
- **logging_steps**: Determines how often to log training information.
- **learning_rate**: The initial learning rate for training, with 1e-4 as a starting point.
- **fp16**: Enables mixed-precision training to reduce memory consumption.
- **max_grad_norm**: The maximum gradient norm for gradient clipping, preventing exploding gradients.
- **max_steps**: The total number of training steps.
- **warmup_ratio**: The proportion of training to perform linear learning rate warmup.
- **group_by_length**: Enables grouping of training data by length for more efficient padding.
- **lr_scheduler_type**: The type of learning rate scheduler to use, with options like "cosine" for cosine learning rate decay.

#### Key Considerations for Fine-Tuning
- Adjusting hyperparameters such as `learning_rate`, and `lr_scheduler_type` based on initial training outcomes can further optimize the fine-tuning process for better model performance.
- The choice between "cosine" and "constant" `learning rate scheduler` can impact the model's learning trajectory and final performance. Experimenting with both can help identify the most effective approach for your specific task and dataset.

## 3. Saving the Model
```python
model.push_to_hub('yourHF/mobile_llama_2kEpoch')
tokenizer.push_to_hub('yourHF/mobile_llama_2kEpoch')
```

The final part of the finetuning script is used to save to [HuggingFace](https://huggingface.co).

## 4. Evaluation
[Evaluation directory](evaluation/llama_bgp_eval_test.ipynb) contains evaluation scripts. The script begins by loading the fine-tuned LLaMA model, preparing it for the evaluation process. It includes procedures to feed prompts to the model and the model's response based on the prompts. In addition, the script contains specialized code to evaluate the model's knowledge of BGP. This evaluation focuses on the accuracy of the model's responses to BGP-related prompts.

# RAG fine-tuning