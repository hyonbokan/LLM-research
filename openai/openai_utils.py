import copy
import dataclasses
import json
import logging
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Union

import tqdm
from openai import OpenAI
from openai import OpenAIError
from dotenv import load_dotenv

# Create a single client instance
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    raise ValueError("No OPENAI_API_KEY found in environment variables or .env!")

client = OpenAI(api_key=api_key)

@dataclasses.dataclass
class OpenAIChatDecodingArguments:
    """
    A set of decoding arguments for the Chat Completions endpoint.
    Adjust defaults to taste or add other parameters (presence_penalty, etc.) as needed.
    """
    temperature: float = 0.7
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[List[str]] = None
    max_tokens: int = 2048
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


def openai_chat_completion(
    messages_batch: Union[List[List[Dict[str, str]]], List[Dict[str, str]]],
    decoding_args: OpenAIChatDecodingArguments,
    model_name: str = "gpt-4o",
    sleep_time: int = 2,
    batch_size: int = 1,
    max_instances: int = sys.maxsize,
    return_text: bool = False,
    **decoding_kwargs,
) -> Union[List[Dict], List[str]]:
    """
    A convenience function for batching OpenAI *Chat* Completions with optional retries and logging.

    Args:
        messages_batch:
            Either:
             - A single list of messages (for a single prompt).
             - A list of lists, where each sub-list is a set of messages for one prompt.
        decoding_args:
            An OpenAIChatDecodingArguments object with model params (temperature, top_p, max_tokens, etc.).
        model_name:
            The Chat model name, e.g. "gpt-4o".
        sleep_time:
            Number of seconds to sleep if we hit a rate limit / error.
        batch_size:
            Number of prompt-message-lists per batch request to the API.
        max_instances:
            Truncates messages_batch to this size if you have many prompts.
        return_text:
            If True, return only the 'content' field from the assistant's message.
        decoding_kwargs:
            Additional kwargs passed to `client.chat.completions.create()`.

    Returns:
        - A list of responses (each response is a `ChatCompletion` object).
    """
    # 1) Detect if single prompt (a single list of dicts)
    is_single_prompt = False
    if messages_batch and isinstance(messages_batch[0], dict):
        messages_batch = [messages_batch]  # Wrap in a list to make it a list of conversations
        is_single_prompt = True

    # 2) Truncate to max_instances if needed
    messages_batch = messages_batch[:max_instances]

    # 3) Split into mini-batches based on batch_size
    num_prompts = len(messages_batch)
    prompt_batches = [
        messages_batch[i * batch_size : (i + 1) * batch_size]
        for i in range(math.ceil(num_prompts / batch_size))
    ]

    completions = []
    decoding_args_dict = dataclasses.asdict(decoding_args)
    decoding_args_dict.update(decoding_kwargs)

    for batch_id, prompt_batch in tqdm.tqdm(
        enumerate(prompt_batches),
        desc="prompt_batches",
        total=len(prompt_batches),
    ):
        batch_responses = []
        for single_conversation in prompt_batch:
            while True:
                try:
                    # Debugging: Print the messages structure
                    # print(f"Sending messages: {json.dumps(single_conversation, indent=2)}")

                    response = client.chat.completions.create(
                        model=model_name,
                        messages=single_conversation,  # A single conversation (list of dicts)
                        **decoding_args_dict,
                    )
                    batch_responses.append(response)
                    break  # Success, exit the retry loop
                except OpenAIError as e:
                    logging.warning(f"OpenAIError: {e}")
                    if "Please reduce your prompt" in str(e):
                        # Reduce max_tokens by 20%
                        decoding_args_dict["max_tokens"] = int(decoding_args_dict["max_tokens"] * 0.8)
                        logging.warning(
                            f"Reducing max_tokens to {decoding_args_dict['max_tokens']}, retrying..."
                        )
                        if decoding_args_dict["max_tokens"] < 1:
                            logging.error("max_tokens reduced below 1, aborting.")
                            return completions
                    else:
                        logging.warning("Hit request rate limit or other error; retrying...")
                        time.sleep(sleep_time)

        completions.extend(batch_responses)

    # If single prompt was initially provided, flatten the list
    if is_single_prompt:
        completions = completions[0]  # Return the single response directly

    return completions

def jdump(obj, path, mode="w", indent=4, default=str):
    """Dump JSON or raw string to a file path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode) as f:
        if isinstance(obj, (dict, list)):
            json.dump(obj, f, indent=indent, default=default)
        elif isinstance(obj, str):
            f.write(obj)
        else:
            raise ValueError(f"Unexpected type to jdump: {type(obj)}")
    f.close()

def jload(path, mode="r"):
    """Load JSON from a file path."""
    with open(path, mode) as f:
        return json.load(f)