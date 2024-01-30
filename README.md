# LLaMA fine-tuning

## 1. Self-instruct Framework
The Self-instruct Framework relies on two main components for generating effective training data: prompts and manual seed tasks. 

For a detailed understanding and background of the Self-instruct Framework, refer to the paper available in the repository: [Self-Instruct Framework Paper](/reference/self-instruct.pdf).

### Prompt and Manual Seed Tasks

- **Prompts**: These serve as guiding directives for the model, ensuring that the generated outputs are not only relevant but also diverse and in alignment with specific criteria. Prompts play a crucial role in instructing and controlling Chat-GPT during the instruction generation process. They set clear parameters and expectations, dictating the nature and scope of the model's output in response to varied scenarios or tasks. 

For practical examples of how prompts are utilized in specific contexts, explore different prompts used for BGP-LLaMA. These include prompts for [general BGP knowledge](/finetuning_dataset/BGP/prompt_knowledge.txt), [use of PyBGPStream library](/finetuning_dataset/BGP/prompt_pybgpstream.txt), and [PyBGPStream real-time analysis](/finetuning_dataset/BGP/prompt_pybgpstream_realtime.txt).

- **Manual Seed Tasks**: These are manually crafted tasks that supply the model with examples of the desired output. They are crucial for instructing the model on how to appropriately respond to the prompts. Just like prompts, seed tasks are created with a particular function in mind. In the case of BGP-LLaMA, you can find seed tasks made for different BGP knowledge areas and functionalities in [/finetuning_dataset/BGP/](/finetuning_dataset/BGP/).

### Running the Self-instruct Code

- **utils.py**: The script manages the connection to OpenAI's GPT models and handles the communication process. The script manages decoding arguments, rate-limiting with sleep intervals, batching of prompts for efficiency, and the handling of API responses. 
`IMPORTANT!`: After the OpenAI API updates in 2024, older models are deprecated. Thus, `utils.py` must be updated to align with the current API standards and model availability.

- **generate_instruction.py**: The main script for the self-instruct framework. Before running it, you need to modify the script to specify the directory paths for the [prompts](/images/self_instruct1.png) and [seed tasks](/images/self_instruct2.png). The script then loads, encodes these components, and generates new sets of instructions. 
To execute the script, use the following command line example, ensuring to replace `--model_name` with the current model you intend to use and adjust `--num_instructions_to_generate` as needed:

```shell
python -m generate_instruction generate_instruction_following_data --num_instructions_to_generate=1000 --model_name="text-davinci-003"
```

## 2. Instruction Fine-tuning

### Installing Dependencies

Start by installing all necessary dependencies listed in `requirements.txt` to ensure your environment is properly set up for the fine-tuning process.

```shell
pip install -r requirements.txt
```

### Fine-tuning Script
The primary script for instruction fine-tuning is located in this [directory](/finetune_main). [Here](/images/finetune_params.png) are the most important hyperparameters.

For VRAM-efficient training, we use Low-Rank Adaptation (LoRA) with the hyperparameters: lora_alpha = 16; lora_dropout = 0.1; lora_r = 64. Keep `lora_alpha` under 32 to maintain the model's optimal performance post fine-tuning. 

Experiment with the `learning rate scheduler`, trying both `"cosine"` and `"constant"` to determine the best outcome. Begin training with these settings and adjust as necessary based on the model's performance.
The final part of the finetuning script is used to save to [HuggingFace](https://huggingface.co).

## 3. Evaluation

[Evaluation directory](evaluation/llama_bgp_eval_test.ipynb) contains evaluation scripts. The script begins by loading the fine-tuned LLaMA model, preparing it for the evaluation process. It includes procedures to feed prompts to the model and the model's response based on the prompts. In addition, the script contains specialized code to evaluate the model's knowledge of BGP. This evaluation focuses on the accuracy of the model's responses to BGP-related prompts.
