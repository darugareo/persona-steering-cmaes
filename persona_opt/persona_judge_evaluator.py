"""
Persona-Aware Judge Evaluator

Wrapper for evaluating responses using the persona-aware judge system.
Loads generated judge prompts and calls OpenAI API for evaluation.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, List
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def _save_raw_judge_log(
    persona_id: str,
    seed: Optional[int],
    method_name: str,
    prompt: str,
    baseline_response: str,
    steered_response: str,
    judge_prompt: str,
    judge_result: Dict,
    layer: Optional[int] = None,
    alpha: Optional[float] = None,
    weights: Optional[List[float]] = None,
    experiment: Optional[str] = None  # New: for Phase 2 experiments (e.g., "truthfulqa", "mmlu")
):
    """Save raw judge log to JSONL file."""
    # Create output directory
    if experiment:
        # Phase 2: reports/raw_judge_logs/{persona_id}/phase2/{experiment}/
        if seed is not None:
            log_dir = Path(f"reports/raw_judge_logs/{persona_id}/phase2/{experiment}")
        else:
            log_dir = Path(f"reports/raw_judge_logs/{persona_id}/phase2/{experiment}")
    else:
        # Phase 1: reports/raw_judge_logs/{persona_id}/seed{seed}/
        if seed is not None:
            log_dir = Path(f"reports/raw_judge_logs/{persona_id}/seed{seed}")
        else:
            log_dir = Path(f"reports/raw_judge_logs/{persona_id}/no_seed")

    log_dir.mkdir(parents=True, exist_ok=True)

    # Prepare log entry
    log_entry = {
        "prompt": prompt,
        "baseline_response": baseline_response,
        "steered_response": steered_response,
        "judge_input": judge_prompt,
        "judge_output": judge_result,
        "meta": {
            "method": method_name,
            "persona_id": persona_id,
            "seed": seed,
            "layer": layer,
            "alpha": alpha,
            "weights": weights
        }
    }

    # Append to JSONL file
    if experiment and seed is not None:
        log_file = log_dir / f"{method_name}_seed{seed}.jsonl"
    else:
        log_file = log_dir / f"{method_name}.jsonl"

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    print(f"  → Saved judge log: {log_file}")


def evaluate_with_persona_judge(
    persona_id: str,
    prompt: str,
    response_a: str,
    response_b: str,
    trait_name: str = "Overall Persona Fit",
    trait_direction: str = "matches persona style and values",
    base_dir: str = "personas",
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_retries: int = 3,
    # New parameters for logging
    method_name: Optional[str] = None,
    seed: Optional[int] = None,
    layer: Optional[int] = None,
    alpha: Optional[float] = None,
    weights: Optional[List[float]] = None,
    save_raw_log: bool = True,
    experiment: Optional[str] = None  # New: Phase 2 experiment name
) -> Dict:
    """
    Evaluate two responses using persona-aware judge.

    Args:
        persona_id: Persona identifier (e.g., "episode-184019_A")
        prompt: User prompt that generated the responses
        response_a: First response to evaluate
        response_b: Second response to evaluate
        trait_name: Name of trait being evaluated
        trait_direction: Direction of trait (e.g., "other-focused")
        base_dir: Base directory containing persona folders
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        model: OpenAI model to use
        temperature: Sampling temperature
        max_retries: Maximum number of retry attempts on API errors

    Returns:
        {
            "winner": "A" or "B" or "tie",
            "confidence": 1-5,
            "persona_fit_score_a": 1-5,
            "persona_fit_score_b": 1-5,
            "explanation": "..."
        }

    Raises:
        FileNotFoundError: If judge prompt template doesn't exist
        ValueError: If API response is invalid
    """
    # Load judge prompt template
    persona_dir = Path(base_dir) / persona_id
    judge_template_path = persona_dir / "final_judge_prompt.txt"

    if not judge_template_path.exists():
        raise FileNotFoundError(
            f"Judge prompt not found: {judge_template_path}\n"
            f"Run: python main_generate_judge.py --base_dir {base_dir} --persona_id {persona_id}"
        )

    judge_template = judge_template_path.read_text(encoding="utf-8")

    # Format judge prompt with evaluation parameters
    judge_prompt = judge_template.format(
        trait_name=trait_name,
        trait_direction=trait_direction,
        prompt=prompt,
        response_a=response_a,
        response_b=response_b
    )

    # Set up OpenAI client
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

    client = openai.OpenAI(api_key=api_key)

    # Call OpenAI API with retries
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a persona-aware evaluation model. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": judge_prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=500
            )

            # Parse JSON response
            result_text = response.choices[0].message.content
            result = json.loads(result_text)

            # Validate required fields
            required_fields = ["winner", "confidence", "persona_fit_score_a", "persona_fit_score_b", "explanation"]
            missing_fields = [f for f in required_fields if f not in result]
            if missing_fields:
                raise ValueError(f"Missing required fields in response: {missing_fields}")

            # Validate field values
            if result["winner"] not in ["A", "B", "tie"]:
                raise ValueError(f"Invalid winner value: {result['winner']}")

            if not (1 <= result["confidence"] <= 5):
                raise ValueError(f"Invalid confidence value: {result['confidence']}")

            if not (1 <= result["persona_fit_score_a"] <= 5):
                raise ValueError(f"Invalid persona_fit_score_a: {result['persona_fit_score_a']}")

            if not (1 <= result["persona_fit_score_b"] <= 5):
                raise ValueError(f"Invalid persona_fit_score_b: {result['persona_fit_score_b']}")

            # Add metadata
            result["model"] = model
            result["persona_id"] = persona_id
            result["trait_name"] = trait_name
            result["trait_direction"] = trait_direction

            # Save raw judge log if requested
            if save_raw_log and method_name:
                _save_raw_judge_log(
                    persona_id=persona_id,
                    seed=seed,
                    method_name=method_name,
                    prompt=prompt,
                    baseline_response=response_a,
                    steered_response=response_b,
                    judge_prompt=judge_prompt,
                    judge_result=result,
                    layer=layer,
                    alpha=alpha,
                    weights=weights,
                    experiment=experiment
                )

            return result

        except (openai.APIError, openai.RateLimitError, json.JSONDecodeError) as e:
            last_error = e
            if attempt < max_retries - 1:
                # Exponential backoff
                import time
                wait_time = 2 ** attempt
                print(f"API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise ValueError(f"Failed after {max_retries} attempts: {e}") from e

    # Should not reach here, but just in case
    raise ValueError(f"Evaluation failed: {last_error}")


def batch_evaluate(
    persona_id: str,
    evaluations: List[Dict[str, str]],
    base_dir: str = "personas",
    **kwargs
) -> List[Dict]:
    """
    Evaluate multiple prompt-response pairs in batch.

    Args:
        persona_id: Persona identifier
        evaluations: List of dicts with keys: "prompt", "response_a", "response_b"
                    Optional: "trait_name", "trait_direction"
        base_dir: Base directory containing persona folders
        **kwargs: Additional arguments passed to evaluate_with_persona_judge

    Returns:
        List of evaluation results
    """
    results = []

    for i, eval_data in enumerate(evaluations):
        print(f"Evaluating {i+1}/{len(evaluations)}...")

        result = evaluate_with_persona_judge(
            persona_id=persona_id,
            prompt=eval_data["prompt"],
            response_a=eval_data["response_a"],
            response_b=eval_data["response_b"],
            trait_name=eval_data.get("trait_name", "Overall Persona Fit"),
            trait_direction=eval_data.get("trait_direction", "matches persona style and values"),
            base_dir=base_dir,
            **kwargs
        )

        results.append(result)

    return results


def compute_aggregate_metrics(results: List[Dict]) -> Dict:
    """
    Compute aggregate metrics from batch evaluation results.

    Args:
        results: List of evaluation results from batch_evaluate

    Returns:
        {
            "win_rate_b": float,  # Win rate for response B
            "tie_rate": float,
            "avg_persona_fit_a": float,
            "avg_persona_fit_b": float,
            "persona_fit_improvement": float,  # B - A
            "avg_confidence": float,
            "total_evaluations": int
        }
    """
    if not results:
        return {}

    total = len(results)

    wins_b = sum(1 for r in results if r["winner"] == "B")
    ties = sum(1 for r in results if r["winner"] == "tie")

    avg_fit_a = sum(r["persona_fit_score_a"] for r in results) / total
    avg_fit_b = sum(r["persona_fit_score_b"] for r in results) / total
    avg_confidence = sum(r["confidence"] for r in results) / total

    return {
        "win_rate_b": wins_b / total,
        "tie_rate": ties / total,
        "avg_persona_fit_a": avg_fit_a,
        "avg_persona_fit_b": avg_fit_b,
        "persona_fit_improvement": avg_fit_b - avg_fit_a,
        "avg_confidence": avg_confidence,
        "total_evaluations": total
    }


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate responses with persona-aware judge")
    parser.add_argument("--persona-id", required=True, help="Persona ID (e.g., episode-184019_A)")
    parser.add_argument("--prompt", required=True, help="User prompt")
    parser.add_argument("--response-a", required=True, help="First response")
    parser.add_argument("--response-b", required=True, help="Second response")
    parser.add_argument("--trait-name", default="Overall Persona Fit", help="Trait name")
    parser.add_argument("--trait-direction", default="matches persona", help="Trait direction")
    parser.add_argument("--base-dir", default="personas", help="Base directory")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")

    args = parser.parse_args()

    result = evaluate_with_persona_judge(
        persona_id=args.persona_id,
        prompt=args.prompt,
        response_a=args.response_a,
        response_b=args.response_b,
        trait_name=args.trait_name,
        trait_direction=args.trait_direction,
        base_dir=args.base_dir,
        model=args.model
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
