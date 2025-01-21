import re
import json
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging
import os


def load_model():
    """
    Load the LLM model and tokenizer.
    """
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    hf_auth = os.environ.get('hf_token')

    # Load model configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map='auto',
        use_auth_token=hf_auth
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    tokenizer.padding_side = "right"

    print("Model and tokenizer loaded successfully")
    return model, tokenizer


def query_llm(model, tokenizer, prompt):
    """
    Queries the LLM with a given prompt and returns the generated output.
    """
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    # Configure generation arguments
    generation_kwargs = dict(
        max_new_tokens=1012,
        do_sample=True,
        temperature=0.70,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    result = pipe(prompt, **generation_kwargs)
    return result[0]['generated_text']


def extract_code_from_reply(llm_output):
    """
    Extracts Python code from LLM output.
    """
    code_pattern = r"```python\s*\n(.*?)```"
    match = re.search(code_pattern, llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def evaluate_code(code):
    """
    Evaluates the extracted code for syntax and runtime errors.

    Returns:
    - dict: Result containing 'status' and 'error_message' if applicable.
    """
    try:
        # Prepare a safe execution environment
        safe_globals = {
            "__builtins__": __builtins__,  # Basic built-ins only
            "__name__": "__main__",       # Mimic script execution
        }
        exec(code, safe_globals)
        return {"status": "pass"}
    except SyntaxError as se:
        return {
            "status": "fail",
            "error_message": f"SyntaxError: {se.msg} at line {se.lineno}, column {se.offset}",
        }
    except Exception as e:
        return {
            "status": "fail",
            "error_message": f"RuntimeError: {traceback.format_exc()}",
        }


def process_prompts(file_path):
    """
    Processes a JSON file with multiple prompts and evaluates each.

    Returns:
    - dict: Summary results including total, passed, and failed counts.
    """
    with open(file_path, 'r') as f:
        instructions = json.load(f)

    model, tokenizer = load_model()
    total = len(instructions)
    passed = 0

    for idx, instruction in enumerate(instructions, start=1):
        name = instruction.get("name", f"Instruction {idx}")
        prompt = instruction["instruction"]

        print(f"\nProcessing '{name}'...")

        # Query the LLM
        llm_output = query_llm(model, tokenizer, prompt)

        # Extract code
        code = extract_code_from_reply(llm_output)
        if not code:
            print(f"FAIL: No Python code block found in the LLM's response.")
            continue

        # Evaluate code
        result = evaluate_code(code)
        if result["status"] == "pass":
            print(f"PASS: Code executed successfully.")
            passed += 1
        else:
            print(f"FAIL: {result['error_message']}")

    # Output summary
    print(f"\nSummary: {passed}/{total} instructions passed.")
    return {"total": total, "passed": passed, "failed": total - passed}


if __name__ == "__main__":
    # Path to JSON file with instructions
    json_file_path = "/home/hb/LLM-research/evaluation/BGP/BGP_analysis_test.json"

    # Process all prompts
    summary = process_prompts(json_file_path)
    print("\nAutomation Complete:")
    print(f"Total: {summary['total']}, Passed: {summary['passed']}, Failed: {summary['failed']}")