#!/usr/bin/env python3
"""
Evaluate steering vectors across multiple layers (layer sweep).

This script:
1. Loads steering vectors for each layer
2. Tests them with different alpha values
3. Compares steered vs baseline responses
4. Uses persona-aware judge to evaluate fit
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import json
import argparse
from typing import List, Dict, Optional
from persona_opt.internal_steering_l3 import Llama3ActivationSteerer
import openai
from tqdm import tqdm
import time


# Sample evaluation prompts for Self-Other Focus trait
EVAL_PROMPTS = [
    "A friend is going through a difficult time. What's your approach?",
    "You have some free time this weekend. How do you decide what to do?",
    "Someone disagrees with your opinion strongly. How do you respond?",
    "You're working on a team project. What's most important to you?",
    "A stranger asks for help on the street. What do you do?",
    "You receive criticism about your work. How do you handle it?",
    "There's one piece of cake left. What happens?",
    "You're planning a vacation. What's your priority?",
    "Someone is talking about their problems. How do you engage?",
    "You have to choose between a personal goal and helping someone. What do you prioritize?",
]


def load_eval_prompts(
    prompts_file: Optional[Path] = None,
    max_prompts: int = 10
) -> List[str]:
    """
    Load evaluation prompts.

    Args:
        prompts_file: Optional path to custom prompts file
        max_prompts: Maximum number of prompts to use

    Returns:
        List of prompt strings
    """
    if prompts_file and prompts_file.exists():
        # Load from file if provided
        with open(prompts_file) as f:
            data = json.load(f)
            if "prompts" in data:
                # Format like prompts_v2.json
                prompts = [p["text"] for p in data["prompts"][:max_prompts]]
            else:
                # Simple list format
                prompts = data[:max_prompts]
        return prompts
    else:
        # Use built-in prompts
        return EVAL_PROMPTS[:max_prompts]


def judge_persona_fit(
    response: str,
    trait_name: str,
    trait_direction: str,
    api_key: str,
    model: str = "gpt-4o-mini"
) -> Dict:
    """
    Use GPT-4o-mini to judge how well a response fits the target persona trait.

    Args:
        response: Assistant's response to evaluate
        trait_name: Name of the trait (e.g., "Self-Other Focus")
        trait_direction: Direction of trait (e.g., "other-focused")
        api_key: OpenAI API key
        model: Judge model to use

    Returns:
        Dictionary with score and explanation
    """
    client = openai.OpenAI(api_key=api_key)

    # Create judge prompt
    judge_prompt = f"""You are evaluating an AI assistant's response for personality traits.

Trait: {trait_name}
Target Direction: {trait_direction}

Response to evaluate:
{response}

Please evaluate how well this response demonstrates {trait_direction} characteristics on a scale of 1-5:
1 = Not at all {trait_direction}
2 = Slightly {trait_direction}
3 = Moderately {trait_direction}
4 = Very {trait_direction}
5 = Extremely {trait_direction}

Respond in JSON format:
{{
  "score": <1-5>,
  "explanation": "<brief explanation of your rating>"
}}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at evaluating personality traits in text."},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        print(f"Warning: Judge API call failed: {e}")
        return {"score": 0, "explanation": f"Error: {str(e)}"}


def compare_responses(
    baseline_response: str,
    steered_response: str,
    trait_name: str,
    trait_direction: str,
    api_key: str,
    model: str = "gpt-4o-mini"
) -> Dict:
    """
    Compare baseline vs steered response and determine which is more aligned.

    Args:
        baseline_response: Response without steering
        steered_response: Response with steering
        trait_name: Name of the trait
        trait_direction: Target direction
        api_key: OpenAI API key
        model: Judge model

    Returns:
        Dictionary with winner and explanation
    """
    client = openai.OpenAI(api_key=api_key)

    compare_prompt = f"""You are comparing two AI assistant responses.

Trait: {trait_name}
Target: We want responses that are more {trait_direction}

Response A:
{baseline_response}

Response B:
{steered_response}

Which response better demonstrates {trait_direction} characteristics?

Respond in JSON format:
{{
  "winner": "A" or "B" or "tie",
  "confidence": <1-5>,
  "explanation": "<brief explanation>"
}}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at comparing personality traits in text."},
                {"role": "user", "content": compare_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        print(f"Warning: Compare API call failed: {e}")
        return {"winner": "tie", "confidence": 0, "explanation": f"Error: {str(e)}"}


def evaluate_layer_sweep(
    model_name: str,
    trait_name: str,
    trait_direction: str,
    layers: List[int],
    alphas: List[float],
    steering_dir: Path,
    eval_prompts: List[str],
    output_path: Path,
    openai_api_key: str,
    device: str = "cuda:0",
    judge_model: str = "gpt-4o-mini",
    max_new_tokens: int = 128,
):
    """
    Evaluate steering vectors across layers and alpha values.

    Args:
        model_name: HuggingFace model name
        trait_name: Trait being evaluated
        trait_direction: Target direction (e.g., "other-focused")
        layers: List of layers to evaluate
        alphas: List of alpha values to test
        steering_dir: Directory containing steering vectors
        eval_prompts: Prompts to use for evaluation
        output_path: Path to save results
        openai_api_key: OpenAI API key for judge
        device: Device to use
        judge_model: Judge model name
        max_new_tokens: Max tokens to generate
    """
    print(f"\n{'='*60}")
    print(f"Layer Sweep Evaluation")
    print(f"Trait: {trait_name} -> {trait_direction}")
    print(f"Layers: {layers}")
    print(f"Alphas: {alphas}")
    print(f"Eval prompts: {len(eval_prompts)}")
    print(f"{'='*60}\n")

    results = {
        "config": {
            "model_name": model_name,
            "trait_name": trait_name,
            "trait_direction": trait_direction,
            "layers": layers,
            "alphas": alphas,
            "num_prompts": len(eval_prompts),
            "judge_model": judge_model,
        },
        "layer_results": {}
    }

    # Process each layer
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Evaluating Layer {layer}")
        print(f"{'='*60}\n")

        # Load steering vector
        steering_path = steering_dir / f"l3_layer{layer:02d}_{trait_name}.pt"
        if not steering_path.exists():
            print(f"Warning: Steering vector not found at {steering_path}, skipping...")
            continue

        steering_vector = torch.load(steering_path)
        print(f"Loaded steering vector: {steering_path}")
        print(f"Vector norm: {steering_vector.norm().item():.4f}")

        # Initialize steerer
        steerer = Llama3ActivationSteerer(
            model_name=model_name,
            target_layer=layer,
            device=device
        )

        layer_results = {
            "layer": layer,
            "steering_path": str(steering_path),
            "alpha_results": {}
        }

        # Evaluate each alpha
        for alpha in alphas:
            print(f"\n--- Alpha = {alpha} ---")

            alpha_results = {
                "alpha": alpha,
                "prompt_results": [],
                "scores": {
                    "baseline_mean": 0.0,
                    "steered_mean": 0.0,
                    "steered_improvement": 0.0,
                },
                "comparisons": {
                    "steered_wins": 0,
                    "baseline_wins": 0,
                    "ties": 0,
                    "win_rate": 0.0,
                }
            }

            # Evaluate on each prompt
            for i, prompt in enumerate(tqdm(eval_prompts, desc=f"Layer {layer}, alpha={alpha}")):
                # Generate baseline (no steering)
                steerer.register_hooks(steering_vector=None)
                baseline_response = steerer.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0
                )

                # Generate with steering
                steerer.register_hooks(
                    steering_vector=steering_vector,
                    alpha=alpha
                )
                steered_response = steerer.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0
                )

                # Judge responses
                baseline_score = judge_persona_fit(
                    baseline_response,
                    trait_name,
                    trait_direction,
                    openai_api_key,
                    judge_model
                )

                steered_score = judge_persona_fit(
                    steered_response,
                    trait_name,
                    trait_direction,
                    openai_api_key,
                    judge_model
                )

                # Compare responses
                comparison = compare_responses(
                    baseline_response,
                    steered_response,
                    trait_name,
                    trait_direction,
                    openai_api_key,
                    judge_model
                )

                # Record results
                prompt_result = {
                    "prompt": prompt,
                    "baseline_response": baseline_response,
                    "steered_response": steered_response,
                    "baseline_score": baseline_score,
                    "steered_score": steered_score,
                    "comparison": comparison,
                }
                alpha_results["prompt_results"].append(prompt_result)

                # Update running stats
                if comparison["winner"] == "B":
                    alpha_results["comparisons"]["steered_wins"] += 1
                elif comparison["winner"] == "A":
                    alpha_results["comparisons"]["baseline_wins"] += 1
                else:
                    alpha_results["comparisons"]["ties"] += 1

                # Rate limiting
                time.sleep(0.5)

            # Compute aggregate scores
            baseline_scores = [r["baseline_score"]["score"] for r in alpha_results["prompt_results"] if r["baseline_score"]["score"] > 0]
            steered_scores = [r["steered_score"]["score"] for r in alpha_results["prompt_results"] if r["steered_score"]["score"] > 0]

            if baseline_scores:
                alpha_results["scores"]["baseline_mean"] = sum(baseline_scores) / len(baseline_scores)
            if steered_scores:
                alpha_results["scores"]["steered_mean"] = sum(steered_scores) / len(steered_scores)
                alpha_results["scores"]["steered_improvement"] = (
                    alpha_results["scores"]["steered_mean"] - alpha_results["scores"]["baseline_mean"]
                )

            # Compute win rate
            total_comparisons = len(eval_prompts)
            alpha_results["comparisons"]["win_rate"] = (
                alpha_results["comparisons"]["steered_wins"] / total_comparisons
            )

            # Print summary
            print(f"\nResults for alpha={alpha}:")
            print(f"  Baseline score: {alpha_results['scores']['baseline_mean']:.2f}")
            print(f"  Steered score:  {alpha_results['scores']['steered_mean']:.2f}")
            print(f"  Improvement:    {alpha_results['scores']['steered_improvement']:+.2f}")
            print(f"  Win rate:       {alpha_results['comparisons']['win_rate']:.1%}")
            print(f"  Wins/Losses/Ties: {alpha_results['comparisons']['steered_wins']}/{alpha_results['comparisons']['baseline_wins']}/{alpha_results['comparisons']['ties']}")

            layer_results["alpha_results"][f"alpha_{alpha}"] = alpha_results

        # Clean up
        steerer.remove_hooks()
        del steerer
        torch.cuda.empty_cache()

        results["layer_results"][f"layer_{layer}"] = layer_results

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")

    # Print summary table
    print("\nSummary Table:")
    print(f"{'Layer':<8} {'Alpha':<8} {'Baseline':<10} {'Steered':<10} {'Improve':<10} {'Win Rate':<10}")
    print("-" * 70)
    for layer in layers:
        layer_key = f"layer_{layer}"
        if layer_key not in results["layer_results"]:
            continue
        for alpha in alphas:
            alpha_key = f"alpha_{alpha}"
            if alpha_key not in results["layer_results"][layer_key]["alpha_results"]:
                continue
            res = results["layer_results"][layer_key]["alpha_results"][alpha_key]
            print(
                f"{layer:<8} {alpha:<8.1f} "
                f"{res['scores']['baseline_mean']:<10.2f} "
                f"{res['scores']['steered_mean']:<10.2f} "
                f"{res['scores']['steered_improvement']:<+10.2f} "
                f"{res['comparisons']['win_rate']:<10.1%}"
            )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate steering vectors across layers"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--trait",
        type=str,
        default="self_other",
        help="Trait name"
    )
    parser.add_argument(
        "--trait-direction",
        type=str,
        default="other-focused",
        help="Target direction for trait"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[20, 21, 22, 23, 24],
        help="Layers to evaluate"
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5],
        help="Alpha values to test"
    )
    parser.add_argument(
        "--steering-dir",
        type=Path,
        default=Path("data/steering_vectors"),
        help="Directory containing steering vectors"
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Optional custom prompts file"
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=10,
        help="Maximum number of prompts to use"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/layer_sweep_l3_self_other.json"),
        help="Output path for results"
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Judge model to use"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max tokens to generate"
    )

    args = parser.parse_args()

    # Get OpenAI API key
    import os
    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required (--openai-api-key or OPENAI_API_KEY env var)")

    # Load evaluation prompts
    eval_prompts = load_eval_prompts(args.prompts_file, args.max_prompts)

    # Run evaluation
    evaluate_layer_sweep(
        model_name=args.model_name,
        trait_name=args.trait,
        trait_direction=args.trait_direction,
        layers=args.layers,
        alphas=args.alphas,
        steering_dir=args.steering_dir,
        eval_prompts=eval_prompts,
        output_path=args.output,
        openai_api_key=api_key,
        device=args.device,
        judge_model=args.judge_model,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
