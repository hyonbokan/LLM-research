import os
import random
import re
import string
import time
import json
import fire
import tqdm
from multiprocessing import Pool
from functools import partial

import numpy as np
from rouge_score import rouge_scorer

from openai_utils import (
    OpenAIChatDecodingArguments,
    openai_chat_completion,
    jdump,
    jload,
)

def encode_prompt_for_chat(prompt_instructions, system_text="/home/hb/LLM-research/dataset/BGP/instruct_finetune_seeds/prompt_pybgpstream.txt"):
    """
    Encode multiple prompt instructions into a chat message list.
    """
    # 1) Load or define your system/developer role messages
    with open(system_text, "r") as f:
        system_role_content = f.read()

    # The chat API expects a list of dictionaries with roles, e.g.:
    messages = [
        {"role": "system", "content": system_role_content},
    ]

    # 2) Then add user instructions as needed. For example:
    user_content = ""
    for idx, d in enumerate(prompt_instructions, start=1):
        inst = re.sub(r"\s+", " ", d["instruction"]).strip()
        inp = d["input"] if d["input"].strip() else "<noinput>"
        out = d["output"]
        user_content += f"\n###\n{idx}. Instruction: {inst}\n{idx}. Input:\n{inp}\n{idx}. Output:\n{out}"

    user_content += f"\n###\n{idx + 1}. Instruction:"  # prompt the assistant to continue

    messages.append({"role": "user", "content": user_content})
    return messages


def post_process_chat_response(num_prompt_instructions, response):
    """
    Parse the assistant's response text and extract new instruction/input/output blocks.
    Returns a list of valid instructions after applying filtering (length, blacklist, etc.).
    """

    # -- 1) Basic checks and extraction of the raw text --
    if not response:
        return []

    # If using return_text=False, 'response' might be a dict with:
    #   { "message": { "role": "assistant", "content": "..." }, "finish_reason": "stop", ... }
    # If using return_text=True, 'response' is a string.
    if isinstance(response, dict) and "message" in response:
        content = response["message"]["content"]
        finish_reason = response.get("finish_reason", None)
    elif isinstance(response, str):
        content = response
        finish_reason = "stop"
    else:
        return []

    # We prepend the (num_prompt_instructions+1). Instruction: to replicate the prompt format
    raw_instructions = f"{num_prompt_instructions + 1}. Instruction:" + content
    # Then split on "###" to separate each block
    blocks = re.split(r"###", raw_instructions)

    instructions = []
    for idx, block in enumerate(blocks):
        # If we see it's truncated due to max_tokens, skip the last partial block
        if idx == len(blocks) - 1 and finish_reason == "length":
            continue

        # Calculate the current instruction number
        current_id = num_prompt_instructions + idx + 1
        # Expected format:
        # "2. Instruction: ...\n2. Input:\n<inp>\n2. Output:\n<out>"
        splitted_data = re.split(
            rf"{current_id}\.\s+(Instruction|Input|Output):",
            block.strip()
        )
        # Check if splitting was successful
        if len(splitted_data) != 7:
            continue

        # Extract the text fields
        inst_str = splitted_data[2].strip()
        input_str = splitted_data[4].strip()
        output_str = splitted_data[6].strip()

        # Convert "<noinput>" back to empty if used
        if input_str.lower() == "<noinput>":
            input_str = ""

        # -------------- Filtering logic --------------
        # (1) Filter out instructions that are too short or too long
        tokens = inst_str.split()
        if len(tokens) <= 3 or len(tokens) > 150:
            continue

        # (2) Blacklist certain keywords
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]

        # (3) Filter "Write a program" style instructions if unwanted
        if inst_str.lower().startswith("write a program"):
            continue

        # (4) Filter those starting with punctuation or non-ASCII
        if inst_str and (inst_str[0] in string.punctuation or not inst_str[0].isascii()):
            continue

        # If it passed all filters, add to the final list
        instructions.append(
            {
                "instruction": inst_str,
                "input": input_str,
                "output": output_str,
            }
        )

    return instructions



def generate_instruction_following_data(
    output_dir="./generated_instructions",
    seed_tasks_path="/home/hb/LLM-research/dataset/BGP/instruct_finetune_seeds/seed_tasks_pybpgstream_legacy.jsonl",
    num_instructions_to_generate=50,
    model_name="gpt-4o",
    num_prompt_instructions=1,
    request_batch_size=2,
    temperature=0.7,
    top_p=1.0,
    num_cpus=4,
    system_text="/home/hb/LLM-research/dataset/BGP/instruct_finetune_seeds/prompt_pybgpstream.txt",
):
    """
    Generate instruction-following data by prompting GPT with seed instructions
    and post-processing the results to remove duplicates or undesirable instructions.
    """
    # 1) Load seed tasks (human-written)
    with open(seed_tasks_path, "r") as f:
        lines = f.read().strip().splitlines()
    seed_tasks = [json.loads(l) for l in lines]

    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    # 2) Prepare output
    os.makedirs(output_dir, exist_ok=True)
    machine_instruction_data = []
    regen_path = os.path.join(output_dir, "regen.json")
    if os.path.exists(regen_path):
        machine_instruction_data = jload(regen_path)
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # 3) Setup a rouge scorer for similarity checks
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # 4) Pre-tokenize existing instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    progress_bar.update(len(machine_instruction_data))

    # 5) Main loop
    request_idx = 0
    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        # Build this batch of messages
        messages_batch = []
        for _ in range(request_batch_size):
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            messages = encode_prompt_for_chat(prompt_instructions, system_text=system_text)
            messages_batch.append(messages)

        # 6) Call the chat completion
        decoding_args = OpenAIChatDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=1200,
            top_p=top_p,
        )

        request_start = time.time()
        results = openai_chat_completion(
            messages_batch=messages_batch,
            decoding_args=decoding_args,
            model_name=model_name,
            batch_size=1,  # Since we are making individual API calls per conversation
            return_text=False,  # Get full response
        )
        request_duration = time.time() - request_start

        # 7) Post-process each result
        process_start = time.time()
        new_instruction_data = []
        for res in results:
            # Extract the assistant's message content
            content = res.choices[0].message.content
            finish_reason = res.choices[0].finish_reason
            
            # Prepare a mock response dict to match the expected input of post_process_chat_response
            mock_response = {
                "message": {"content": content},
                "finish_reason": finish_reason
            }
            
            # Parse out new instructions
            parsed_instructions = post_process_chat_response(num_prompt_instructions, mock_response)
            new_instruction_data += parsed_instructions

        total = len(new_instruction_data)
        keep = 0
        for entry in new_instruction_data:
            # Similarity check
            new_tokens = scorer._tokenizer.tokenize(entry["instruction"])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            if max(rouge_scores) > 0.7:
                continue  # Too similar to an existing instruction
            keep += 1

            # Save the new instruction
            entry["most_similar_instructions"] = {
                all_instructions[i]: rouge_scores[i]
                for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            entry["avg_similarity_score"] = float(np.mean(rouge_scores))

            machine_instruction_data.append(entry)
            all_instructions.append(entry["instruction"])
            all_instruction_tokens.append(new_tokens)
            progress_bar.update(1)

        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")

        # Save progress
        jdump(machine_instruction_data, regen_path)

    progress_bar.close()

def main(task="generate_instruction_following_data", **kwargs):
    if task == "generate_instruction_following_data":
        generate_instruction_following_data(**kwargs)
    else:
        print("No such task:", task)


if __name__ == "__main__":
    fire.Fire(main)