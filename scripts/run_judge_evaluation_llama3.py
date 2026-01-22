#!/usr/bin/env python3
"""
Judge Evaluation for 10 Personas (Llama-3 Same-Model Evaluation)

Design:
- All 10 personas: base vs equal, base vs prompt
- All personas with optimized: equal vs optimized, base vs optimized
- Judge: gpt-4o-mini (main) + 25% gpt-4o (spot check)

This evaluates Llama-3-8B generated outputs to test if optimized weights
work when the same model is used for optimization and evaluation.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import random

import openai
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
ALL_PERSONAS = [
    "episode-184019_A",
    "episode-239427_A",
    "episode-118328_B",
    "episode-5289_A",
    "episode-29600_A",
    "episode-88279_B",
    "episode-132247_A",
    "episode-166805_A",
    "episode-196697_B",
    "episode-225888_A",
]

TARGET_MODEL = "llama3_8b"
INPUT_BASE = Path(f"results/same_model/{TARGET_MODEL}")
OUTPUT_FILE = Path("results/judge_evaluation/10personas_llama3_results.json")
LOG_FILE = Path("logs/judge_evaluation_llama3.log")

JUDGE_MODEL_MAIN = "gpt-4o-mini"
JUDGE_MODEL_SPOT = "gpt-4o"
SPOT_CHECK_RATIO = 0.25


def get_optimized_personas():
    """Dynamically detect which personas have optimized.jsonl"""
    optimized_personas = []
    for persona_id in ALL_PERSONAS:
        optimized_path = INPUT_BASE / persona_id / "optimized.jsonl"
        if optimized_path.exists():
            optimized_personas.append(persona_id)
    return optimized_personas


# Dynamically build comparison pairs based on available files
OPTIMIZED_PERSONAS = get_optimized_personas()

COMPARISON_PAIRS = [
    ("base", "equal", ALL_PERSONAS),           # All 10 personas
    ("base", "prompt", ALL_PERSONAS),          # All 10 personas
    ("equal", "optimized", OPTIMIZED_PERSONAS), # All with optimized
    ("base", "optimized", OPTIMIZED_PERSONAS),  # All with optimized
]


def log(msg, console=True):
    """Log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"

    if console:
        print(log_msg)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(log_msg + '\n')


def load_responses(persona_id: str, method: str) -> List[Dict]:
    """Load generated responses for a persona and method."""
    file_path = INPUT_BASE / persona_id / f"{method}.jsonl"

    if not file_path.exists():
        return []

    responses = []
    with open(file_path, 'r') as f:
        for line in f:
            responses.append(json.loads(line))

    return responses


def judge_pair(prompt: str, response_a: str, response_b: str, judge_model: str) -> Dict:
    """Judge a pair of responses using LLM."""
    system_prompt = """You are an expert evaluator comparing two AI responses.
Compare the responses based on:
1. Relevance to the question
2. Accuracy and correctness
3. Clarity and coherence
4. Helpfulness

Respond in JSON format:
{
  "winner": "A" or "B" or "tie",
  "reason": "brief explanation in English"
}"""

    user_prompt = f"""Question: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Output JSON only."""

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        result["judge_model"] = judge_model
        return result

    except Exception as e:
        log(f"  ✗ Judge error: {e}")
        return {"winner": "error", "reason": str(e), "judge_model": judge_model}


def evaluate_pair(method_a: str, method_b: str, personas: List[str]) -> Dict:
    """Evaluate one comparison pair across personas."""
    log(f"\nEvaluating: {method_a} vs {method_b} ({len(personas)} personas)")

    results = []
    total_comparisons = 0
    spot_checks = 0

    for persona_id in tqdm(personas, desc=f"{method_a} vs {method_b}"):
        # Load responses
        responses_a = load_responses(persona_id, method_a)
        responses_b = load_responses(persona_id, method_b)

        if not responses_a:
            log(f"  ⚠️  {persona_id}/{method_a}: no data")
            continue

        if not responses_b:
            log(f"  ⚠️  {persona_id}/{method_b}: no data")
            continue

        # Match prompts
        for resp_a in responses_a:
            prompt_id = resp_a["prompt_id"]
            # Handle both field name formats
            prompt_text = resp_a.get("prompt") or resp_a.get("prompt_text", "")
            response_a_text = resp_a.get("response") or resp_a.get("response_text", "")

            # Find matching response from B
            resp_b = next((r for r in responses_b if r["prompt_id"] == prompt_id), None)

            if not resp_b:
                continue

            response_b_text = resp_b.get("response") or resp_b.get("response_text", "")

            # Decide judge model (25% spot check with gpt-4o)
            use_spot_check = random.random() < SPOT_CHECK_RATIO
            judge_model = JUDGE_MODEL_SPOT if use_spot_check else JUDGE_MODEL_MAIN

            if use_spot_check:
                spot_checks += 1

            # Judge
            judgment = judge_pair(
                prompt_text,
                response_a_text,
                response_b_text,
                judge_model,
            )

            results.append({
                "persona_id": persona_id,
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "method_a": method_a,
                "method_b": method_b,
                "response_a": response_a_text,
                "response_b": response_b_text,
                "winner": judgment["winner"],
                "reason": judgment.get("reason", ""),
                "judge_model": judge_model,
            })

            total_comparisons += 1

    # Compute win rates
    wins_a = sum(1 for r in results if r["winner"] == "A")
    wins_b = sum(1 for r in results if r["winner"] == "B")
    ties = sum(1 for r in results if r["winner"] == "tie")
    errors = sum(1 for r in results if r["winner"] == "error")

    win_rate_a = wins_a / total_comparisons if total_comparisons > 0 else 0
    win_rate_b = wins_b / total_comparisons if total_comparisons > 0 else 0

    log(f"  Comparisons: {total_comparisons}")
    log(f"  Spot checks (gpt-4o): {spot_checks} ({spot_checks/total_comparisons*100:.1f}%)")
    log(f"  {method_a} wins: {wins_a} ({win_rate_a*100:.1f}%)")
    log(f"  {method_b} wins: {wins_b} ({win_rate_b*100:.1f}%)")
    log(f"  Ties: {ties}")
    log(f"  Errors: {errors}")

    return {
        "method_a": method_a,
        "method_b": method_b,
        "personas": personas,
        "num_personas": len(personas),
        "total_comparisons": total_comparisons,
        "spot_checks": spot_checks,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "errors": errors,
        "win_rate_a": win_rate_a,
        "win_rate_b": win_rate_b,
        "judgments": results,
    }


def main():
    """Run judge evaluation for 10 personas."""
    log("="*80)
    log("JUDGE EVALUATION: 10 PERSONAS (LLAMA-3 SAME-MODEL)")
    log("="*80)
    log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Personas: {len(ALL_PERSONAS)}")
    log(f"  With optimized: {len(OPTIMIZED_PERSONAS)}")
    log(f"Main judge: {JUDGE_MODEL_MAIN}")
    log(f"Spot check: {JUDGE_MODEL_SPOT} ({SPOT_CHECK_RATIO*100:.0f}%)")
    log("="*80)

    # Set random seed for reproducibility
    random.seed(42)

    # Check API key
    import os
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        log("✗ ERROR: OpenAI API key not set")
        log("  export OPENAI_API_KEY=your_key")
        return 1

    log(f"✓ OpenAI API key found ({len(api_key)} characters)")

    # Verify input files
    log("\nVerifying input files...")
    missing = []

    for persona_id in ALL_PERSONAS:
        persona_dir = INPUT_BASE / persona_id
        for method in ["base", "prompt", "equal"]:
            file_path = persona_dir / f"{method}.jsonl"
            if not file_path.exists():
                missing.append(f"{persona_id}/{method}")

    # Check optimized for all personas that have it
    for persona_id in OPTIMIZED_PERSONAS:
        file_path = INPUT_BASE / persona_id / "optimized.jsonl"
        if not file_path.exists():
            missing.append(f"{persona_id}/optimized")

    if missing:
        log(f"✗ ERROR: {len(missing)} file(s) missing:")
        for m in missing[:10]:
            log(f"  - {m}")
        if len(missing) > 10:
            log(f"  ... and {len(missing)-10} more")
        return 1

    log("✓ All input files found")
    log(f"  Personas with optimized: {len(OPTIMIZED_PERSONAS)}/10")

    # Estimate comparisons
    total_comparisons = (
        len(ALL_PERSONAS) * 28 * 2 +        # base vs equal, base vs prompt
        len(OPTIMIZED_PERSONAS) * 28 * 2    # equal vs optimized, base vs optimized
    )

    log(f"\nEstimated comparisons: {total_comparisons}")
    log(f"Estimated API calls: {int(total_comparisons * (1 - SPOT_CHECK_RATIO))} (gpt-4o-mini) + {int(total_comparisons * SPOT_CHECK_RATIO)} (gpt-4o)")
    log(f"Estimated time: {total_comparisons * 2 / 3600:.1f} hours")
    log("\nStarting evaluation...")

    # Run evaluations
    start_time = datetime.now()
    all_results = []

    for method_a, method_b, personas in COMPARISON_PAIRS:
        result = evaluate_pair(method_a, method_b, personas)
        all_results.append(result)

    # Save results
    output_data = {
        "date": datetime.now().isoformat(),
        "model": "llama3_8b",
        "evaluation_type": "same_model",
        "personas": {
            "all": ALL_PERSONAS,
            "with_optimized": OPTIMIZED_PERSONAS,
        },
        "judge_models": {
            "main": JUDGE_MODEL_MAIN,
            "spot_check": JUDGE_MODEL_SPOT,
            "spot_check_ratio": SPOT_CHECK_RATIO,
        },
        "results": all_results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    elapsed = (datetime.now() - start_time).total_seconds() / 3600

    log("\n" + "="*80)
    log("EVALUATION COMPLETE")
    log("="*80)
    log(f"Elapsed time: {elapsed:.2f} hours")
    log(f"Output: {OUTPUT_FILE}")
    log(f"Log: {LOG_FILE}")

    # Summary
    log("\nSummary:")
    for result in all_results:
        log(f"  {result['method_a']} vs {result['method_b']} ({result['num_personas']} personas):")
        log(f"    {result['method_b']} wins: {result['win_rate_b']*100:.1f}%")

    log("\nNext step: Aggregate results")
    log("  python scripts/aggregate_results_llama3.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
