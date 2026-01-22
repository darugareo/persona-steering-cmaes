"""
GPT-OSS Wrapper for Python 3.8 -> Python 3.12 Bridge

This module provides a wrapper to use gpt-oss-20b from Python 3.8 by calling
the conda environment with Python 3.12 where gpt-oss is installed.
"""

import json
import subprocess
import sys
from typing import Dict, List

CONDA_ENV_PYTHON = "/home/nakata/anaconda3/envs/gpt_oss/bin/python"


def generate_with_gpt_oss(
    messages: List[Dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """
    Generate text using gpt-oss-20b via conda environment

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text string
    """
    # Create a Python script to run in the conda environment
    script = f'''
import json
import sys
import os
from transformers import pipeline
import torch

# Set cache directory to user-writable location
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface")

model_id = "openai/gpt-oss-20b"

# Load pipeline
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Get input from stdin
input_data = json.loads(sys.stdin.read())
messages = input_data["messages"]
max_new_tokens = input_data["max_new_tokens"]
temperature = input_data["temperature"]

# Generate
outputs = pipe(
    messages,
    max_new_tokens=max_new_tokens,
    do_sample=temperature > 0,
    temperature=temperature if temperature > 0 else 1.0,
)

# Extract generated text
generated_text = outputs[0]["generated_text"][-1]["content"]

# Output as JSON
print(json.dumps({{"text": generated_text}}))
'''

    # Prepare input data
    input_data = {
        "messages": messages,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }

    try:
        # Set environment variables for subprocess
        import os
        env = os.environ.copy()
        env["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
        env["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface")

        # Run the script in the conda environment
        result = subprocess.run(
            [CONDA_ENV_PYTHON, "-c", script],
            input=json.dumps(input_data),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes timeout (increased for long texts)
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(f"gpt-oss generation failed: {result.stderr}")

        # Parse output
        output_data = json.loads(result.stdout.strip().split('\n')[-1])  # Last line is JSON
        return output_data["text"]

    except Exception as e:
        raise RuntimeError(f"Error calling gpt-oss wrapper: {e}")


def test_wrapper():
    """Test the wrapper"""
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]

    print("Testing gpt-oss wrapper...")
    result = generate_with_gpt_oss(messages, max_new_tokens=50)
    print(f"Result: {result}")


if __name__ == "__main__":
    test_wrapper()
