import re
import json
import traceback
import os
from threading import Lock
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging

# Constants
CUSTOM_MODEL = "hyonbokan/BGPStream13-10k-cutoff-1024-max-2048"
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_LOCK = Lock()

# Global model and tokenizer
model = None
tokenizer = None


def load_model():
    """
    Load the LLM model and tokenizer in a thread-safe manner.
    """
    global model, tokenizer

    if model is None or tokenizer is None:
        with MODEL_LOCK:  # Prevents race conditions
            if model is None or tokenizer is None:
                try:
                    model_id = CUSTOM_MODEL
                    hf_auth = os.environ.get('HF_TOKEN')

                    print(f"Loading model: {model_id}")
                    
                    # Load model
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        device_map='auto',
                        use_auth_token=hf_auth
                    )

                    # Load tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.padding_side = "left"
                    tokenizer.truncation_side = "left"

                    print("Model and tokenizer loaded successfully")
                except Exception as e:
                    print(f"Error loading model: {str(e)}")
                    raise
    return model, tokenizer


def query_llm(prompt):
    """
    Queries the LLM with a given prompt and returns the generated output.
    """
    # print(f"\nProcessing query: {prompt}")
    
    try:
        model, tokenizer = load_model()

        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1500
        )

        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        # Generation settings
        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1012,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Generate response
        generated_ids = model.generate(**generation_kwargs)

        # Decode response
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("Generated text:\n")
        print(generated_text)
        print("\n")
        return generated_text

    except Exception as e:
        print(f"Error during LLM query: {str(e)}")
        return None


def extract_code_from_reply(llm_output):
    """
    Extracts Python code from LLM output.
    """
    code_pattern = r"```(?:\w+)?\s*\n(.*?)```"
    match = re.search(code_pattern, llm_output, re.DOTALL)
    return match.group(1).strip() if match else None


def evaluate_code(code):
    """
    Evaluates extracted code for syntax and runtime errors.
    Returns a dictionary with execution status.
    """
    try:
        safe_globals = {
            "__builtins__": __builtins__,
            "__name__": "__main__",
        }
        exec(code, safe_globals)
        return {"status": "pass"}
    
    except SyntaxError as se:
        return {"status": "fail", "error_message": f"SyntaxError: {se.msg} at line {se.lineno}, column {se.offset}"}
    
    except Exception as e:
        return {"status": "fail", "error_message": f"RuntimeError: {traceback.format_exc()}"}


def process_prompts(file_path):
    """
    Processes a JSON file containing multiple prompts and evaluates each.
    Returns a detailed summary of pass/fail results, including level-wise counts.
    """
    try:
        with open(file_path, 'r') as f:
            instructions = json.load(f)

        model, tokenizer = load_model()
        total = len(instructions)
        passed = 0

        # Level-wise tracking
        level_counts = {1: 0, 2: 0, 3: 0}   # Total count of each level
        level_passed = {1: 0, 2: 0, 3: 0}   # Count of passed tests per level
        passed_tasks = []                   # Store task names that passed

        for idx, instruction in enumerate(instructions, start=1):
            name = instruction.get("task_name", f"Instruction {idx}")
            prompt = instruction["instruction"]
            level = instruction.get("level", 1)  # Default to level 1 if missing

            # Track the number of tasks at each level
            if level in level_counts:
                level_counts[level] += 1

            print(f"\nProcessing '{name}' (Level {level})...")

            # Query the LLM
            llm_output = query_llm(prompt)

            if not llm_output:
                print(f"FAIL: No response from LLM.")
                continue

            # Extract code
            code = extract_code_from_reply(llm_output)
            if not code:
                print(f"FAIL: No Python code block found.")
                continue

            # Evaluate code
            result = evaluate_code(code)
            if result["status"] == "pass":
                print(f"PASS: Code executed successfully.")
                passed += 1
                passed_tasks.append(name)  # Store the task name that passed

                # Track level-wise pass count
                if level in level_passed:
                    level_passed[level] += 1
            else:
                print(f"FAIL: {result['error_message']}")

        # Output summary
        print("\n=== Evaluation Summary ===")
        print(f"Total Tests: {total}, Passed: {passed}, Failed: {total - passed}")

        for lvl in level_counts:
            print(f"Level {lvl}: {level_passed[lvl]}/{level_counts[lvl]} passed")

        print("\nPassed Tasks:")
        for task in passed_tasks:
            print(f" - {task}")

        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "level_summary": {
                "level_1": {"total": level_counts[1], "passed": level_passed[1]},
                "level_2": {"total": level_counts[2], "passed": level_passed[2]},
                "level_3": {"total": level_counts[3], "passed": level_passed[3]},
            },
            "passed_tasks": passed_tasks
        }

    except Exception as e:
        print(f"Error processing prompts: {str(e)}")
        return {"total": 0, "passed": 0, "failed": 0, "level_summary": {}, "passed_tasks": []}


if __name__ == "__main__":
    json_file_path = "/home/hb/LLM-research/evaluation/BGP/BGP_analysis_test.json"
    summary = process_prompts(json_file_path)

    print("\n=== Final Summary ===")
    print(f"Total: {summary['total']}, Passed: {summary['passed']}, Failed: {summary['failed']}")
    print("Level-wise Summary:")
    for level, stats in summary["level_summary"].items():
        print(f"{level}: {stats['passed']}/{stats['total']} passed")

    print("\nList of Passed Tasks:")
    for task in summary["passed_tasks"]:
        print(f" - {task}")