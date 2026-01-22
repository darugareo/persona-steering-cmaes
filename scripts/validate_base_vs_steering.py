#!/usr/bin/env python3
"""
Base vs Steering-only comparison experiment.

This validates whether activation steering alone (without persona prompts)
has any effect compared to the unmodified base model.

Conditions:
- Base: No steering, no persona prompt (just answer the question)
- Steering-only: Optimized steering vectors, no persona prompt

This is critical to decompose:
- Prompt effect: Base vs Prompt-only
- Steering effect: Base vs Steering-only
- Hybrid effect: Base vs Hybrid
"""

import torch
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.internal_steering_l3 import Llama3ActivationSteerer
from persona_opt.persona_judge_evaluator import evaluate_with_persona_judge


class BaseSteeringValidator:
    """
    Validates Base vs Steering-only performance.

    Base: Model generates response with no modifications
    Steering-only: Model with optimized steering vectors but no persona prompt
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        layer: int = 20,
        alpha: float = 2.0,
        judge_model: str = "gpt-4o",
        device: str = "cuda:0"
    ):
        self.layer = layer
        self.alpha = alpha
        self.judge_model = judge_model
        self.device = device

        print(f"\n{'='*80}")
        print(f"Base vs Steering-only Validation")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Layer: {layer}")
        print(f"Alpha: {alpha}")
        print(f"Judge: {judge_model}")
        print(f"Device: {device}")
        print(f"{'='*80}\n")

        # Initialize steerer
        self.steerer = Llama3ActivationSteerer(
            model_name=model_name,
            target_layer=layer,
            device=device
        )

        # Load trait vectors
        self.trait_vectors = self._load_trait_vectors()

    def _load_trait_vectors(self) -> Dict[str, torch.Tensor]:
        """Load SVD trait vectors for steering."""
        trait_dir = Path("data/steering_vectors_v2")
        trait_names = ["R1", "R2", "R3", "R4", "R5"]

        vectors = {}
        for trait in trait_names:
            vector_path = trait_dir / trait / f"layer{self.layer}_svd.pt"
            if not vector_path.exists():
                raise FileNotFoundError(f"Trait vector not found: {vector_path}")

            # Load and move to device immediately
            vector = torch.load(vector_path, map_location='cpu', weights_only=False)
            vectors[trait] = vector.to(self.device)

        print(f"✓ Loaded {len(vectors)} trait vectors")
        return vectors

    def _load_persona_weights(self, persona_id: str) -> Dict[str, float]:
        """Load optimized trait weights for a persona."""
        results_dir = Path("optimization_results_26personas")

        # Try both gpu directories
        for gpu_dir in ["gpu0", "gpu1"]:
            weights_file = results_dir / gpu_dir / f"{persona_id}_best_weights.json"
            if weights_file.exists():
                with open(weights_file) as f:
                    weights = json.load(f)
                    return weights  # File contains weights directly

        raise FileNotFoundError(f"Weights not found for {persona_id}")

    def generate_base(self, prompt: str) -> str:
        """
        Generate response with base model (no modifications).

        Just answer the question directly without persona.
        """
        # Remove any hooks
        self.steerer.remove_hooks()

        # Generate with simple user prompt (no persona injection)
        response = self.steerer.generate(
            prompt=prompt,
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False
        )

        # Extract only generated part (skip prompt)
        # Format the prompt to match what the model sees
        if self.steerer.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.steerer.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # Remove the formatted prompt from response
        if response.startswith(formatted_prompt):
            response = response[len(formatted_prompt):].strip()

        return response

    def generate_steering_only(
        self,
        prompt: str,
        trait_weights: Dict[str, float]
    ) -> str:
        """
        Generate response with steering only (no persona prompt).

        Apply optimized trait vectors but use same simple prompt as base.
        """
        # Register steering hooks
        self.steerer.register_hooks(
            multi_trait_vectors=self.trait_vectors,
            trait_weights=trait_weights
        )

        # Generate (same as base, just with steering active)
        response = self.steerer.generate(
            prompt=prompt,
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False
        )

        # Extract only generated part
        if self.steerer.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.steerer.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        if response.startswith(formatted_prompt):
            response = response[len(formatted_prompt):].strip()

        # Remove hooks
        self.steerer.remove_hooks()

        return response

    def compare_pair(
        self,
        persona_id: str,
        prompt: str,
        response_base: str,
        response_steering: str
    ) -> Dict:
        """
        Use LLM judge to compare Base vs Steering-only.

        Judge which response better matches the persona profile.
        """
        import random

        # Randomize position to avoid bias
        steering_is_a = random.choice([True, False])

        if steering_is_a:
            response_a = response_steering
            response_b = response_base
        else:
            response_a = response_base
            response_b = response_steering

        # Get judge evaluation
        result = evaluate_with_persona_judge(
            persona_id=persona_id,
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            trait_name="Overall Persona Fit",
            trait_direction="matches persona style and values",
            model=self.judge_model,
            temperature=0.3,
            save_raw_log=False
        )

        # Map judge winner to our conditions
        judge_winner = result.get("winner", "tie")

        if judge_winner == "A":
            winner = "steering" if steering_is_a else "base"
        elif judge_winner == "B":
            winner = "base" if steering_is_a else "steering"
        else:
            winner = "tie"

        return {
            "persona_id": persona_id,
            "prompt": prompt,
            "response_base": response_base,
            "response_steering": response_steering,
            "steering_is_a": steering_is_a,
            "judge_winner": judge_winner,
            "winner": winner,
            "confidence": result.get("confidence", 0),
            "explanation": result.get("explanation", "")
        }

    def run_validation(
        self,
        personas: List[Dict],
        prompts: List[str],
        output_dir: Path,
        resume_from: Optional[Path] = None
    ) -> Dict:
        """
        Run full Base vs Steering-only validation.

        Args:
            personas: List of persona dicts with id and text
            prompts: List of evaluation prompts
            output_dir: Directory to save results
            resume_from: Optional checkpoint to resume from
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "comparison_results.json"

        # Load checkpoint if resuming
        completed = set()
        results = []

        if resume_from and resume_from.exists():
            print(f"Resuming from {resume_from}")
            with open(resume_from) as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data.get("results", [])
                for r in results:
                    completed.add((r["persona_id"], r["prompt"]))
            print(f"Loaded {len(results)} completed comparisons")

        # Run comparisons
        total = len(personas) * len(prompts)

        with tqdm(total=total, desc="Comparing Base vs Steering") as pbar:
            for persona in personas:
                persona_id = persona["id"]

                # Load optimized weights
                try:
                    trait_weights = self._load_persona_weights(persona_id)
                except FileNotFoundError as e:
                    print(f"\nWarning: {e}")
                    pbar.update(len(prompts))
                    continue

                for prompt in prompts:
                    # Skip if already done
                    if (persona_id, prompt) in completed:
                        pbar.update(1)
                        continue

                    # Generate base response
                    response_base = self.generate_base(prompt)

                    # Generate steering-only response
                    response_steering = self.generate_steering_only(
                        prompt=prompt,
                        trait_weights=trait_weights
                    )

                    # Judge comparison
                    comparison = self.compare_pair(
                        persona_id=persona_id,
                        prompt=prompt,
                        response_base=response_base,
                        response_steering=response_steering
                    )

                    results.append(comparison)
                    pbar.update(1)

                    # Checkpoint every 10 comparisons
                    if len(results) % 10 == 0:
                        checkpoint = {
                            "date": datetime.now().isoformat(),
                            "config": {
                                "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                                "layer": self.layer,
                                "alpha": self.alpha,
                                "judge_model": self.judge_model
                            },
                            "total_comparisons": len(results),
                            "results": results
                        }
                        with open(results_file, 'w') as f:
                            json.dump(checkpoint, f, indent=2)

        # Final save
        final_data = {
            "date": datetime.now().isoformat(),
            "config": {
                "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                "layer": self.layer,
                "alpha": self.alpha,
                "judge_model": self.judge_model
            },
            "total_comparisons": len(results),
            "results": results
        }

        with open(results_file, 'w') as f:
            json.dump(final_data, f, indent=2)

        print(f"\n✓ Saved results to {results_file}")

        return final_data


def load_personas() -> List[Dict]:
    """Load all 26 optimized personas."""
    # Load from optimization results
    summary_file = Path("optimization_results_26personas/analysis_summary.json")

    if not summary_file.exists():
        raise FileNotFoundError(f"Analysis summary not found: {summary_file}")

    with open(summary_file) as f:
        data = json.load(f)

    personas = [{"id": p["persona_id"]} for p in data["personas"]]
    print(f"✓ Loaded {len(personas)} optimized personas")
    return personas


def load_test_prompts() -> List[str]:
    """Load v2 prompts (28 prompts, unseen during optimization)."""
    prompts_file = Path("data/eval_prompts/persona_eval_prompts_v2.json")

    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    with open(prompts_file) as f:
        data = json.load(f)

    # Extract all prompts from categories
    all_prompts = []
    for category, prompts in data["categories"].items():
        for prompt_obj in prompts:
            all_prompts.append(prompt_obj["text"])

    print(f"✓ Loaded {len(all_prompts)} test prompts")
    return all_prompts


def analyze_results(results_file: Path) -> Dict:
    """Analyze results and generate statistical summary."""
    with open(results_file) as f:
        data = json.load(f)

    results = data["results"]
    total = len(results)

    # Count wins
    wins = {"base": 0, "steering": 0, "tie": 0}
    confidences = []
    per_persona = {}

    for r in results:
        winner = r["winner"]
        wins[winner] += 1
        confidences.append(r["confidence"])

        persona_id = r["persona_id"]
        if persona_id not in per_persona:
            per_persona[persona_id] = {"base": 0, "steering": 0, "tie": 0}
        per_persona[persona_id][winner] += 1

    # Statistics
    import numpy as np
    from scipy import stats

    win_rates = {k: v/total for k, v in wins.items()}
    avg_confidence = np.mean(confidences)

    # Decisive comparisons (exclude ties)
    decisive = wins["steering"] + wins["base"]

    if decisive > 0:
        steering_decisive_rate = wins["steering"] / decisive

        # Binomial test
        try:
            result = stats.binomtest(wins["steering"], n=decisive, p=0.5, alternative='two-sided')
            binom_p = float(result.pvalue)
        except AttributeError:
            binom_p = float(stats.binom_test(wins["steering"], n=decisive, p=0.5, alternative='two-sided'))

        # Cohen's h
        p1 = wins["steering"] / decisive
        p2 = wins["base"] / decisive
        h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

        if abs(h) < 0.2:
            effect_interp = "negligible"
        elif abs(h) < 0.5:
            effect_interp = "small"
        elif abs(h) < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"

        # Wilson CI
        from statsmodels.stats.proportion import proportion_confint
        ci_lower, ci_upper = proportion_confint(
            wins["steering"], decisive, alpha=0.05, method='wilson'
        )
    else:
        steering_decisive_rate = 0.0
        binom_p = 1.0
        h = 0.0
        effect_interp = "undefined"
        ci_lower = 0.0
        ci_upper = 0.0

    summary = {
        "date": data["date"],
        "config": data["config"],
        "total_comparisons": total,
        "wins": wins,
        "win_rates": win_rates,
        "decisive_comparisons": decisive,
        "decisive_win_rates": {
            "steering": float(steering_decisive_rate),
            "base": float(1 - steering_decisive_rate) if decisive > 0 else 0.0
        },
        "statistical_tests": {
            "binomial_test_p_value": binom_p,
            "significant_at_0.05": bool(binom_p < 0.05),
            "cohens_h": float(h),
            "effect_size_interpretation": effect_interp
        },
        "confidence_interval_95": {
            "steering_win_rate_lower": float(ci_lower),
            "steering_win_rate_upper": float(ci_upper)
        },
        "average_judge_confidence": float(avg_confidence),
        "per_persona": per_persona
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Base vs Steering-only validation")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--layer", type=int, default=20, help="Steering layer")
    parser.add_argument("--alpha", type=float, default=2.0, help="Steering strength")
    parser.add_argument("--judge-model", type=str, default="gpt-4o", help="Judge model")
    parser.add_argument("--limit-personas", type=int, help="Limit number of personas (for testing)")
    parser.add_argument("--limit-prompts", type=int, help="Limit number of prompts (for testing)")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint file")

    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"

    # Load data
    print("Loading personas and prompts...")
    personas = load_personas()
    prompts = load_test_prompts()

    if args.limit_personas:
        personas = personas[:args.limit_personas]
        print(f"Limited to {len(personas)} personas")

    if args.limit_prompts:
        prompts = prompts[:args.limit_prompts]
        print(f"Limited to {len(prompts)} prompts")

    print(f"\nTotal comparisons: {len(personas)} × {len(prompts)} = {len(personas) * len(prompts)}")

    # Initialize validator
    validator = BaseSteeringValidator(
        layer=args.layer,
        alpha=args.alpha,
        judge_model=args.judge_model,
        device=device
    )

    # Run validation
    output_dir = Path("results/base_vs_steering")
    resume_from = Path(args.resume) if args.resume else None

    print("\nStarting validation...")
    results_data = validator.run_validation(
        personas=personas,
        prompts=prompts,
        output_dir=output_dir,
        resume_from=resume_from
    )

    # Analyze results
    print("\nAnalyzing results...")
    results_file = output_dir / "comparison_results.json"
    summary = analyze_results(results_file)

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Saved summary to {summary_file}")

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Total comparisons: {summary['total_comparisons']}")
    print(f"\nWin counts:")
    print(f"  Steering: {summary['wins']['steering']} ({summary['win_rates']['steering']*100:.1f}%)")
    print(f"  Base:     {summary['wins']['base']} ({summary['win_rates']['base']*100:.1f}%)")
    print(f"  Tie:      {summary['wins']['tie']} ({summary['win_rates']['tie']*100:.1f}%)")

    if summary['decisive_comparisons'] > 0:
        print(f"\nDecisive comparisons: {summary['decisive_comparisons']}")
        print(f"  Steering win rate: {summary['decisive_win_rates']['steering']*100:.1f}%")
        print(f"\nStatistical significance:")
        print(f"  p-value: {summary['statistical_tests']['binomial_test_p_value']:.4f}")
        print(f"  Significant (p<0.05): {summary['statistical_tests']['significant_at_0.05']}")
        print(f"  Cohen's h: {summary['statistical_tests']['cohens_h']:.3f}")
        print(f"  Effect size: {summary['statistical_tests']['effect_size_interpretation']}")

    print(f"\nAverage judge confidence: {summary['average_judge_confidence']:.2f}/5.0")
    print("="*80)


if __name__ == "__main__":
    main()
