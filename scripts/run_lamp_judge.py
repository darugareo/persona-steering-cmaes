#!/usr/bin/env python3
"""
LaMP-7 LLM-as-a-Judge Evaluation - Phase 2-C

Uses A/B comparison instead of absolute scoring for more reliable evaluation.
Compares generated outputs pairwise to determine which better matches user style.

Judge models: gpt-4o-mini, gpt-4o
User profiles ARE shown to judge (but were NOT shown during generation).
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
import time
from collections import defaultdict
import openai
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


JUDGE_PROMPT_TEMPLATE = """You are evaluating two tweet paraphrases to determine which better matches a specific user's writing style.

# User Profile (24 historical tweets)
{profile_tweets}

# Original Tweet to Paraphrase
{original_tweet}

# Reference Paraphrase (Gold Standard)
{reference}

# Paraphrase A
{paraphrase_a}

# Paraphrase B
{paraphrase_b}

# Task
Compare Paraphrase A and Paraphrase B. Which one better matches the user's writing style shown in their profile tweets?

Consider:
- Tone (formal vs. casual)
- Vocabulary choices
- Punctuation patterns (e.g., ellipsis, capitalization)
- Sentence structure
- Use of emoji or special characters

# Required Output (JSON only)
{{
  "winner": "A" or "B",
  "confidence": 1-5 (1=barely different, 5=very clear difference),
  "explanation": "Brief explanation (1-2 sentences)"
}}"""


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_judge_cache(cache_path: str) -> Dict:
    """Load cached judge responses."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_judge_cache(cache_path: str, cache: Dict):
    """Save judge response cache."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)


def make_cache_key(sample_id: str, method_a: str, method_b: str, judge_model: str) -> str:
    """Create cache key for judge response."""
    return f"{sample_id}_{method_a}_vs_{method_b}_{judge_model}"


def format_profile_tweets(profile: List[Dict[str, str]], max_tweets: int = 24) -> str:
    """Format user profile tweets for prompt."""
    tweets = [f"{i+1}. {t['text']}" for i, t in enumerate(profile[:max_tweets])]
    return "\n".join(tweets)


def extract_original_tweet(input_text: str) -> str:
    """Extract the original tweet from LaMP-7 input format."""
    # Input format: "Paraphrase the following tweet without any explanation before or after it: {tweet}"
    if ":" in input_text:
        return input_text.split(":", 1)[1].strip()
    return input_text


def call_judge(judge_model: str, prompt: str, temperature: float = 0.0) -> Dict:
    """Call OpenAI judge model."""
    try:
        client = openai.OpenAI()  # Uses OPENAI_API_KEY from environment
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        result_text = response.choices[0].message.content
        result = json.loads(result_text)

        # Validate response format
        assert "winner" in result and result["winner"] in ["A", "B"]
        assert "confidence" in result and 1 <= result["confidence"] <= 5
        assert "explanation" in result

        return result
    except Exception as e:
        logger.error(f"Judge API error: {e}")
        return {
            "winner": "A",  # Default
            "confidence": 1,
            "explanation": f"[ERROR: {str(e)}]",
            "error": str(e)
        }


def run_pairwise_comparison(
    sample_id: str,
    profile: List[Dict],
    input_text: str,
    reference: str,
    output_a: str,
    output_b: str,
    method_a: str,
    method_b: str,
    judge_model: str,
    cache: Dict,
    temperature: float = 0.0
) -> Dict:
    """Run A/B comparison for a single pair."""
    cache_key = make_cache_key(sample_id, method_a, method_b, judge_model)

    # Check cache
    if cache_key in cache:
        logger.debug(f"Cache hit for {cache_key}")
        return cache[cache_key]

    # Format prompt
    profile_text = format_profile_tweets(profile)
    original_tweet = extract_original_tweet(input_text)

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        profile_tweets=profile_text,
        original_tweet=original_tweet,
        reference=reference,
        paraphrase_a=output_a,
        paraphrase_b=output_b
    )

    # Call judge
    result = call_judge(judge_model, prompt, temperature)

    # Add metadata
    result['sample_id'] = sample_id
    result['method_a'] = method_a
    result['method_b'] = method_b
    result['judge_model'] = judge_model

    # Cache result
    cache[cache_key] = result

    # Rate limiting
    time.sleep(0.1)  # Be nice to API

    return result


def main():
    parser = argparse.ArgumentParser(description="LaMP-7 A/B Judge Evaluation")

    parser.add_argument("--data-path", type=str, default="data/lamp7",
                       help="Path to LaMP-7 dataset (for profiles)")
    parser.add_argument("--results-dir", type=str, default="outputs/lamp7",
                       help="Directory with generated results")
    parser.add_argument("--output", type=str, default="results/lamp7/judge_comparisons.json",
                       help="Output file for judge results")

    # Judge configuration
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini",
                       choices=["gpt-4o-mini", "gpt-4o"],
                       help="Judge model")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Judge temperature (0 for deterministic)")

    # Comparison pairs
    parser.add_argument("--comparisons", type=str, nargs="+",
                       default=["base_vs_prompt", "base_vs_equal", "base_vs_optimized", "equal_vs_optimized"],
                       help="Comparison pairs (format: methodA_vs_methodB)")

    # Cache
    parser.add_argument("--cache", type=str, default="cache/lamp7_judge_cache.json",
                       help="Cache file path")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of samples (for testing)")

    args = parser.parse_args()

    # Check for API key
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Load cache
    cache = load_judge_cache(args.cache)
    logger.info(f"Loaded cache with {len(cache)} entries")

    # Load generated results
    results_dir = Path(args.results_dir)
    methods_data = {}
    for method in ["base", "prompt", "equal", "optimized"]:
        results_file = results_dir / f"{method}_results.jsonl"
        if results_file.exists():
            methods_data[method] = {item['id']: item for item in load_jsonl(results_file)}
            logger.info(f"Loaded {len(methods_data[method])} samples for {method}")

    # Load LaMP-7 data for profiles
    questions_path = Path(args.data_path) / "dev_questions.json"
    with open(questions_path) as f:
        questions = json.load(f)
    profiles = {q['id']: q['profile'] for q in questions}

    # Parse comparison pairs
    comparisons = []
    for comp in args.comparisons:
        parts = comp.split("_vs_")
        if len(parts) == 2:
            comparisons.append((parts[0], parts[1]))
        else:
            logger.warning(f"Invalid comparison format: {comp}, skipping")

    logger.info(f"Running {len(comparisons)} comparison types")

    # Get sample IDs (intersection of all methods)
    all_sample_ids = set.intersection(*[set(methods_data[m].keys()) for m, _ in comparisons] +
                                       [set(methods_data[m].keys()) for _, m in comparisons])
    sample_ids = sorted(all_sample_ids)[:args.limit]
    logger.info(f"Comparing {len(sample_ids)} samples")

    # Run comparisons
    all_results = []
    for method_a, method_b in comparisons:
        logger.info(f"\nComparing {method_a} vs {method_b}")

        comp_results = []
        for sample_id in tqdm(sample_ids, desc=f"{method_a} vs {method_b}"):
            data_a = methods_data[method_a][sample_id]
            data_b = methods_data[method_b][sample_id]
            profile = profiles[sample_id]

            result = run_pairwise_comparison(
                sample_id=sample_id,
                profile=profile,
                input_text=data_a['input'],
                reference=data_a['gold'],
                output_a=data_a['generated'],
                output_b=data_b['generated'],
                method_a=method_a,
                method_b=method_b,
                judge_model=args.judge_model,
                cache=cache,
                temperature=args.temperature
            )

            comp_results.append(result)

        all_results.extend(comp_results)

        # Save cache after each comparison type
        save_judge_cache(args.cache, cache)
        logger.info(f"Saved cache with {len(cache)} entries")

    # Save all results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'judge_model': args.judge_model,
            'num_samples': len(sample_ids),
            'comparisons': comparisons,
            'results': all_results
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(all_results)} judge results to {output_path}")

    # Quick summary
    for method_a, method_b in comparisons:
        comp_results = [r for r in all_results if r['method_a'] == method_a and r['method_b'] == method_b]
        wins_a = sum(1 for r in comp_results if r['winner'] == 'A')
        wins_b = sum(1 for r in comp_results if r['winner'] == 'B')
        total = len(comp_results)
        logger.info(f"{method_a} vs {method_b}: {wins_a}/{total} ({100*wins_a/total:.1f}%) A wins, {wins_b}/{total} ({100*wins_b/total:.1f}%) B wins")

    logger.info("Judge evaluation complete!")


if __name__ == "__main__":
    main()
