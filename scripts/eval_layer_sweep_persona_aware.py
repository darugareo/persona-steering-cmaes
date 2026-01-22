#!/usr/bin/env python3
"""
Persona-Aware Layer Sweep Evaluation.

This script fixes the critical flaw in eval_layer_sweep_l3.py:
- OLD: Judge based on "generic trait-likeness"
- NEW: Judge based on "fit to SPECIFIC PERSONA"

Key differences:
1. Judge receives persona profile, example responses, behavioral patterns
2. Evaluation measures persona-specific alignment, not generic traits
3. Each evaluation is tied to an individual from ConversationChronicles
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import json
import argparse
from typing import List, Dict, Optional
from persona_opt.internal_steering_l3 import Llama3ActivationSteerer
from persona_opt.persona_aware_judge import (
    create_persona_aware_judge_prompt,
    create_persona_fit_scoring_prompt,
    load_persona_profile
)
from persona_opt.eval_prompts import load_eval_prompts, get_prompts_summary
import openai
from tqdm import tqdm
import time


def judge_with_persona_awareness(
    persona_profile: Dict,
    original_prompt: str,
    response: str,
    api_key: str,
    model: str = "gpt-4o-mini"
) -> Dict:
    """
    Judge response based on PERSONA-SPECIFIC fit.

    Args:
        persona_profile: Full persona profile dictionary
        original_prompt: Original user question
        response: Response to evaluate
        api_key: OpenAI API key
        model: Judge model

    Returns:
        Dictionary with persona_fit_score and explanation
    """
    client = openai.OpenAI(api_key=api_key)

    # Create persona-aware judge prompt
    judge_prompt = create_persona_fit_scoring_prompt(
        persona_profile=persona_profile,
        original_prompt=original_prompt,
        response=response
    )

    try:
        response_obj = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at evaluating persona-specific response alignment."},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        result = json.loads(response_obj.choices[0].message.content)
        return result

    except Exception as e:
        print(f"Warning: Persona-aware judge API call failed: {e}")
        return {"persona_fit_score": 0, "explanation": f"Error: {str(e)}"}


def compare_responses_persona_aware(
    persona_profile: Dict,
    original_prompt: str,
    baseline_response: str,
    steered_response: str,
    api_key: str,
    model: str = "gpt-4o-mini"
) -> Dict:
    """
    Compare baseline vs steered based on persona-specific fit.

    Args:
        persona_profile: Full persona profile
        original_prompt: User question
        baseline_response: Response without steering
        steered_response: Response with steering
        api_key: OpenAI API key
        model: Judge model

    Returns:
        Dictionary with winner, confidence, and persona fit scores
    """
    client = openai.OpenAI(api_key=api_key)

    # Create persona-aware comparison prompt
    compare_prompt = create_persona_aware_judge_prompt(
        persona_profile=persona_profile,
        original_prompt=original_prompt,
        response_a=baseline_response,
        response_b=steered_response,
        comparison_type="persona_fit"
    )

    try:
        response_obj = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at evaluating persona-specific response alignment."},
                {"role": "user", "content": compare_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        result = json.loads(response_obj.choices[0].message.content)
        return result

    except Exception as e:
        print(f"Warning: Persona-aware comparison failed: {e}")
        return {
            "winner": "tie",
            "confidence": 0,
            "explanation": f"Error: {str(e)}",
            "persona_fit_score_a": 0,
            "persona_fit_score_b": 0
        }


def evaluate_persona_aware_steering(
    model_name: str,
    persona_id: str,
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
    Evaluate steering with persona-aware judge.

    Args:
        model_name: HuggingFace model name
        persona_id: Specific persona to evaluate against
        layers: Layers to evaluate
        alphas: Alpha values to test
        steering_dir: Directory with steering vectors
        eval_prompts: Prompts for evaluation
        output_path: Output file path
        openai_api_key: OpenAI API key
        device: Device to use
        judge_model: Judge model
        max_new_tokens: Max tokens to generate
    """
    print(f"\n{'='*60}")
    print(f"Persona-Aware Layer Sweep Evaluation")
    print(f"Persona: {persona_id}")
    print(f"Layers: {layers}")
    print(f"Alphas: {alphas}")
    print(f"Eval prompts: {len(eval_prompts)}")
    print(f"{'='*60}\n")

    # Load persona profile
    print(f"Loading persona profile: {persona_id}")
    persona_profile = load_persona_profile(persona_id)
    print(f"  Sessions: {persona_profile.get('num_sessions', 0)}")
    print(f"  Examples: {len(persona_profile.get('example_responses', []))}")

    results = {
        "config": {
            "model_name": model_name,
            "persona_id": persona_id,
            "evaluation_type": "persona_aware",
            "layers": layers,
            "alphas": alphas,
            "num_prompts": len(eval_prompts),
            "judge_model": judge_model,
        },
        "persona_profile_summary": {
            "num_sessions": persona_profile.get('num_sessions', 0),
            "communication_style": persona_profile.get('communication_style', {}),
            "relationship_contexts": persona_profile.get('relationship_contexts', {}),
        },
        "layer_results": {}
    }

    # Process each layer
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Evaluating Layer {layer}")
        print(f"{'='*60}\n")

        # Load steering vector
        steering_path = steering_dir / f"l3_layer{layer:02d}_self_other.pt"
        if not steering_path.exists():
            print(f"Warning: Steering vector not found, skipping...")
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
                "persona_fit_scores": {
                    "baseline_mean": 0.0,
                    "steered_mean": 0.0,
                    "improvement": 0.0,
                },
                "comparisons": {
                    "steered_wins": 0,
                    "baseline_wins": 0,
                    "ties": 0,
                    "win_rate": 0.0,
                }
            }

            # Evaluate on each prompt
            for prompt in tqdm(eval_prompts, desc=f"Layer {layer}, alpha={alpha}"):
                # Generate baseline
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

                # Persona-aware judging
                baseline_score = judge_with_persona_awareness(
                    persona_profile,
                    prompt,
                    baseline_response,
                    openai_api_key,
                    judge_model
                )

                steered_score = judge_with_persona_awareness(
                    persona_profile,
                    prompt,
                    steered_response,
                    openai_api_key,
                    judge_model
                )

                # Persona-aware comparison
                comparison = compare_responses_persona_aware(
                    persona_profile,
                    prompt,
                    baseline_response,
                    steered_response,
                    openai_api_key,
                    judge_model
                )

                # Record results
                prompt_result = {
                    "prompt": prompt,
                    "baseline_response": baseline_response,
                    "steered_response": steered_response,
                    "baseline_persona_fit": baseline_score,
                    "steered_persona_fit": steered_score,
                    "comparison": comparison,
                }
                alpha_results["prompt_results"].append(prompt_result)

                # Update stats
                if comparison["winner"] == "B":
                    alpha_results["comparisons"]["steered_wins"] += 1
                elif comparison["winner"] == "A":
                    alpha_results["comparisons"]["baseline_wins"] += 1
                else:
                    alpha_results["comparisons"]["ties"] += 1

                # Rate limiting
                time.sleep(0.5)

            # Compute aggregate scores
            baseline_scores = [
                r["baseline_persona_fit"].get("persona_fit_score", 0)
                for r in alpha_results["prompt_results"]
                if r["baseline_persona_fit"].get("persona_fit_score", 0) > 0
            ]
            steered_scores = [
                r["steered_persona_fit"].get("persona_fit_score", 0)
                for r in alpha_results["prompt_results"]
                if r["steered_persona_fit"].get("persona_fit_score", 0) > 0
            ]

            if baseline_scores:
                alpha_results["persona_fit_scores"]["baseline_mean"] = sum(baseline_scores) / len(baseline_scores)
            if steered_scores:
                alpha_results["persona_fit_scores"]["steered_mean"] = sum(steered_scores) / len(steered_scores)
                alpha_results["persona_fit_scores"]["improvement"] = (
                    alpha_results["persona_fit_scores"]["steered_mean"] -
                    alpha_results["persona_fit_scores"]["baseline_mean"]
                )

            # Compute win rate
            total = len(eval_prompts)
            alpha_results["comparisons"]["win_rate"] = alpha_results["comparisons"]["steered_wins"] / total

            # Print summary
            print(f"\nResults for alpha={alpha}:")
            print(f"  Baseline persona fit: {alpha_results['persona_fit_scores']['baseline_mean']:.2f}")
            print(f"  Steered persona fit:  {alpha_results['persona_fit_scores']['steered_mean']:.2f}")
            print(f"  Improvement:          {alpha_results['persona_fit_scores']['improvement']:+.2f}")
            print(f"  Win rate:             {alpha_results['comparisons']['win_rate']:.1%}")
            print(f"  W/L/T: {alpha_results['comparisons']['steered_wins']}/{alpha_results['comparisons']['baseline_wins']}/{alpha_results['comparisons']['ties']}")

            layer_results["alpha_results"][f"alpha_{alpha}"] = alpha_results

        # Cleanup
        steerer.remove_hooks()
        del steerer
        torch.cuda.empty_cache()

        results["layer_results"][f"layer_{layer}"] = layer_results

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")

    # Summary table
    print("\nSummary Table (Persona-Specific Fit):")
    print(f"{'Layer':<8} {'Alpha':<8} {'Baseline':<12} {'Steered':<12} {'Improve':<12} {'Win Rate':<10}")
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
                f"{res['persona_fit_scores']['baseline_mean']:<12.2f} "
                f"{res['persona_fit_scores']['steered_mean']:<12.2f} "
                f"{res['persona_fit_scores']['improvement']:<+12.2f} "
                f"{res['comparisons']['win_rate']:<10.1%}"
            )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Persona-aware layer sweep evaluation"
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model name"
    )
    parser.add_argument(
        "--persona-id",
        required=True,
        help="Persona ID to evaluate against (REQUIRED)"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[20, 22, 24],
        help="Layers to evaluate"
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1.0],
        help="Alpha values"
    )
    parser.add_argument(
        "--steering-dir",
        type=Path,
        default=Path("data/steering_vectors"),
        help="Steering vectors directory"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Evaluation prompts (if not specified, loads from prompts file)"
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=Path("data/eval_prompts/persona_eval_prompts_v1.json"),
        help="Prompts JSON file"
    )
    parser.add_argument(
        "--prompt-category",
        default=None,
        help="If specified, only use prompts from this category"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Limit number of prompts (default: all)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/layer_sweep_persona_aware.json"),
        help="Output path"
    )
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key"
    )
    parser.add_argument(
        "--device",
        default="cuda:0"
    )

    args = parser.parse_args()

    # Get API key
    import os
    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required")

    # Load prompts
    if args.prompts is not None:
        eval_prompts = args.prompts
        print(f"Using {len(eval_prompts)} custom prompts")
    else:
        eval_prompts = load_eval_prompts(
            prompts_file=args.prompts_file,
            category=args.prompt_category,
            num_prompts=args.num_prompts
        )
        summary = get_prompts_summary(args.prompts_file)
        print(f"\nLoaded prompts from: {args.prompts_file}")
        print(f"  Version: {summary['version']}")
        print(f"  Total available: {summary['total_prompts']}")
        print(f"  Using: {len(eval_prompts)} prompts")
        if args.prompt_category:
            print(f"  Category filter: {args.prompt_category}")

    # Run evaluation
    evaluate_persona_aware_steering(
        model_name=args.model_name,
        persona_id=args.persona_id,
        layers=args.layers,
        alphas=args.alphas,
        steering_dir=args.steering_dir,
        eval_prompts=eval_prompts,
        output_path=args.output,
        openai_api_key=api_key,
        device=args.device,
    )


if __name__ == "__main__":
    main()
