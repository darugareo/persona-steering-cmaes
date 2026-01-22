#!/usr/bin/env python3
"""
Evaluate GPT-OSS generation ability across:
  - JSON instruction following
  - Summarization
  - Persona profiling

Pass/Fail criteria included.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.gpt_oss_wrapper import generate_with_gpt_oss

# ───────────────────────────────────
# TEST 1: JSON FORMAT ABILITY
# ───────────────────────────────────
def test_json_output():
    """Test if gpt-oss can follow JSON output instructions"""
    messages = [
        {
            "role": "system",
            "content": "You are a strict JSON output generator. Only output valid JSON."
        },
        {
            "role": "user",
            "content": """Output a JSON object with keys: "user", "score".
Values:
- user: string "nakata"
- score: integer 10
Only output valid JSON. Do not include any explanation."""
        }
    ]

    print("=" * 80)
    print("TEST 1: JSON FORMAT ABILITY")
    print("=" * 80)

    try:
        out = generate_with_gpt_oss(messages, max_new_tokens=100, temperature=0.0)
        print(f"Output:\n{out}\n")

        # Try to parse JSON
        try:
            j = json.loads(out.strip())
            if j.get("user") == "nakata" and j.get("score") == 10:
                print("✓ PASS: Valid JSON with correct values")
                return True, out
            else:
                print(f"✗ FAIL: JSON parsed but values incorrect: {j}")
                return False, out
        except json.JSONDecodeError as e:
            print(f"✗ FAIL: Invalid JSON - {e}")
            return False, out
    except Exception as e:
        print(f"✗ FAIL: Generation error - {e}")
        return False, str(e)

# ───────────────────────────────────
# TEST 2: SUMMARIZATION ABILITY
# ───────────────────────────────────
def test_summary():
    """Test if gpt-oss can summarize text concisely"""
    text = """
I had a terrible day yesterday. The bus was late,
my laptop crashed during work, and I forgot my wallet.
But my friend helped me in the evening, and I felt
much better after talking with them.
"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides concise summaries."
        },
        {
            "role": "user",
            "content": f"Summarize this in one sentence:\n{text}"
        }
    ]

    print("=" * 80)
    print("TEST 2: SUMMARIZATION ABILITY")
    print("=" * 80)

    try:
        out = generate_with_gpt_oss(messages, max_new_tokens=150, temperature=0.3)
        print(f"Output:\n{out}\n")

        # Check if summary is concise (< 200 chars)
        if len(out.strip()) < 200:
            print(f"✓ PASS: Concise summary ({len(out)} chars)")
            return True, out
        else:
            print(f"✗ FAIL: Summary too long ({len(out)} chars)")
            return False, out
    except Exception as e:
        print(f"✗ FAIL: Generation error - {e}")
        return False, str(e)

# ───────────────────────────────────
# TEST 3: PERSONA PROFILING ABILITY
# ───────────────────────────────────
def test_persona_profile():
    """Test if gpt-oss can extract personality traits from conversation"""
    convo = """
A: I've been feeling unsure about my work lately.
B: Why is that?
A: I don't know… I hesitate too much.
B: Maybe try making a small plan first?
A: You're right… I tend to think too negatively.
"""
    messages = [
        {
            "role": "system",
            "content": "You are a personality analyst. Analyze conversations and describe personality traits."
        },
        {
            "role": "user",
            "content": f"""Analyze speaker A and describe their personality in 4–6 traits.
Use natural language. Be specific and evidence-based.

Conversation:
{convo}"""
        }
    ]

    print("=" * 80)
    print("TEST 3: PERSONA PROFILING ABILITY")
    print("=" * 80)

    try:
        out = generate_with_gpt_oss(messages, max_new_tokens=300, temperature=0.5)
        print(f"Output:\n{out}\n")

        # Check if output contains relevant personality traits
        out_lower = out.lower()
        trait_keywords = [
            "hesitant", "negative", "anxious", "passive", "planning",
            "self-doubt", "uncertain", "pessimistic", "indecisive"
        ]

        found_traits = [kw for kw in trait_keywords if kw in out_lower]

        if len(found_traits) >= 2:
            print(f"✓ PASS: Found {len(found_traits)} relevant traits: {found_traits}")
            return True, out
        else:
            print(f"✗ FAIL: Only found {len(found_traits)} relevant traits: {found_traits}")
            return False, out
    except Exception as e:
        print(f"✗ FAIL: Generation error - {e}")
        return False, str(e)

# ───────────────────────────────────
# MAIN
# ───────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("GPT-OSS GENERATION ABILITY EVALUATION")
    print("=" * 80 + "\n")

    results = {}

    # Test 1: JSON
    ok, out = test_json_output()
    results["TEST_JSON"] = ok

    # Test 2: Summary
    ok, out = test_summary()
    results["TEST_SUMMARY"] = ok

    # Test 3: Persona Profile
    ok, out = test_persona_profile()
    results["TEST_PERSONA_PROFILE"] = ok

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")

    print("\n" + "=" * 80)
    if all(results.values()):
        print(">>> ✓ PASS: GPT-OSS is suitable for persona profile generation.")
        print(">>> Recommendation: Use gpt-oss for persona summaries (cost: $0)")
    else:
        print(">>> ✗ FAIL: GPT-OSS cannot be used reliably for persona profile generation.")
        print(">>> Recommendation: Use GPT-4o-mini for persona summaries (cost: ~$0.10-0.50)")
    print("=" * 80 + "\n")

    sys.exit(0 if all(results.values()) else 1)
