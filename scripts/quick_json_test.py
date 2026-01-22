#!/usr/bin/env python3
"""
Quick test: Can gpt-oss output JSON at all?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.gpt_oss_wrapper import generate_with_gpt_oss

# Extremely simple JSON test
prompt = """Output ONLY this JSON object, nothing else:

{"test": 1, "success": true}

Do not write any explanation. Start your response with {"""

messages = [{"role": "user", "content": prompt}]

print("Testing if gpt-oss can output JSON...")
print("(This will take ~30-60 seconds for model loading)\n")

try:
    result = generate_with_gpt_oss(
        messages=messages,
        max_new_tokens=50,
        temperature=0.0,
    )

    print("="*60)
    print("OUTPUT:")
    print("="*60)
    print(result)
    print("="*60)

    # Check if it starts with {
    if result.strip().startswith("{"):
        print("\n✓ Model CAN output JSON (starts with {)")
    else:
        print("\n✗ Model does NOT output JSON (starts with other text)")
        print(f"   First 50 chars: {result[:50]}")

except Exception as e:
    print(f"✗ ERROR: {e}")
