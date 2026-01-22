#!/usr/bin/env python3
"""
Large-Scale Hybrid Validation Experiment
==========================================

Validates Hybrid (Prompt + Steering) vs Prompt-only across:
- 26 personas (all optimized personas)
- 28 prompts from v2 (unused in optimization - proper test set)
- Judge: GPT-4o (different from optimization's GPT-4o-mini)
- Total: 728 pairwise comparisons

This addresses three critical issues:
1. Train/Test Split: Uses v2 prompts (never seen during optimization)
2. Judge Overfitting: Uses GPT-4o instead of GPT-4o-mini
3. Sample Size: 728 comparisons vs previous 10

Usage:
    python scripts/validate_hybrid_large_scale.py --gpu_id 0
"""

import argparse
import json
import random
import sys
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np
from scipy import stats

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.internal_steering_l3 import Llama3ActivationSteerer
from persona_opt.baselines.prompt_persona import PromptPersonaMethod
from persona_opt.persona_judge_evaluator import evaluate_with_persona_judge


class HybridValidator:
    """Large-scale validation of Hybrid vs Prompt-only methods."""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        layer: int = 20,
        alpha: float = 2.0,
        device: str = "cuda:0",
        judge_model: str = "gpt-4o",
        checkpoint_dir: str = "results/hybrid_validation_large_scale"
    ):
        """
        Initialize validator.

        Args:
            model_name: HuggingFace model name
            layer: Layer for steering
            alpha: Steering strength
            device: GPU device
            judge_model: Judge model (GPT-4o to avoid overfitting)
            checkpoint_dir: Directory for saving results
        """
        self.model_name = model_name
        self.layer = layer
        self.alpha = alpha
        self.device = device
        self.judge_model = judge_model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        print(f"\n{'='*80}")
        print(f"LARGE-SCALE HYBRID VALIDATION")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Layer: {layer}, Alpha: {alpha}")
        print(f"Device: {device}")
        print(f"Judge: {judge_model}")
        print(f"{'='*80}\n")

        self.steerer = Llama3ActivationSteerer(
            model_name=model_name,
            target_layer=layer,
            device=device
        )

        # Load trait vectors
        self.trait_vectors = self._load_trait_vectors()

        # Generation params
        self.gen_params = {
            "max_new_tokens": 150,
            "do_sample": False,
            "pad_token_id": self.steerer.tokenizer.eos_token_id
        }

    def _load_trait_vectors(self) -> Dict[str, torch.Tensor]:
        """Load SVD trait vectors for the target layer."""
        trait_dir = Path("data/steering_vectors_v2")
        vectors = {}
        trait_names = ["R1", "R2", "R3", "R4", "R5"]

        for trait in trait_names:
            vector_path = trait_dir / trait / f"layer{self.layer}_svd.pt"
            if not vector_path.exists():
                raise FileNotFoundError(f"Trait vector not found: {vector_path}")
            # Load and move to device immediately
            vector = torch.load(vector_path, map_location='cpu', weights_only=False)
            vectors[trait] = vector.to(self.device)

        print(f"✓ Loaded {len(vectors)} trait vectors")
        return vectors

    def _load_optimized_weights(self, persona_id: str) -> Dict[str, float]:
        """Load optimized weights for a persona."""
        # Check both GPUs
        for gpu_id in [0, 1]:
            weight_file = Path(f"optimization_results_26personas/gpu{gpu_id}/{persona_id}_best_weights.json")
            if weight_file.exists():
                with open(weight_file) as f:
                    return json.load(f)

        raise FileNotFoundError(f"Optimized weights not found for {persona_id}")

    def _load_persona_profile(self, persona_id: str) -> str:
        """Load persona profile text."""
        profile_path = Path(f"personas/{persona_id}/persona_profile.txt")
        if not profile_path.exists():
            raise FileNotFoundError(f"Persona profile not found: {profile_path}")

        with open(profile_path) as f:
            return f.read().strip()

    def _generate_prompt_only(self, persona_id: str, prompt: str) -> str:
        """Generate response with prompt-only method."""
        # Build persona description from profile
        profile_text = self._load_persona_profile(persona_id)
        persona_description = profile_text  # Use full profile

        # Format with system message
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful AI assistant. {persona_description}"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Apply chat template
        formatted_prompt = self.steerer.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Remove hooks for baseline generation
        self.steerer.remove_hooks()

        # Generate
        inputs = self.steerer.tokenizer(
            formatted_prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.steerer.model.generate(
                **inputs,
                **self.gen_params
            )

        # Decode only the generated part (skip input)
        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_length:]
        response = self.steerer.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return response

    def _generate_hybrid(self, persona_id: str, prompt: str, weights: Dict[str, float]) -> str:
        """Generate response with hybrid method (prompt + steering)."""
        # 1. Load persona profile for prompt
        profile_text = self._load_persona_profile(persona_id)
        persona_description = profile_text

        # 2. Register steering hooks
        self.steerer.register_hooks(
            multi_trait_vectors=self.trait_vectors,
            trait_weights=weights
        )

        # 3. Format with system message (same as prompt-only)
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful AI assistant. {persona_description}"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        formatted_prompt = self.steerer.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 4. Generate with steering
        inputs = self.steerer.tokenizer(
            formatted_prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.steerer.model.generate(
                **inputs,
                **self.gen_params
            )

        # 5. Decode only the generated part
        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_length:]
        response = self.steerer.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Remove hooks
        self.steerer.remove_hooks()

        return response

    def _evaluate_pair(
        self,
        persona_id: str,
        prompt: str,
        response_hybrid: str,
        response_prompt: str,
        randomize_order: bool = True
    ) -> Dict:
        """
        Evaluate a pair of responses using GPT-4o judge.

        Args:
            persona_id: Persona identifier
            prompt: Input prompt
            response_hybrid: Hybrid method response
            response_prompt: Prompt-only method response
            randomize_order: Whether to randomize A/B order

        Returns:
            Dict with evaluation results
        """
        # Randomize order to avoid position bias
        if randomize_order and random.random() < 0.5:
            response_a = response_prompt
            response_b = response_hybrid
            hybrid_is_a = False
        else:
            response_a = response_hybrid
            response_b = response_prompt
            hybrid_is_a = True

        # Call judge
        judge_result = evaluate_with_persona_judge(
            persona_id=persona_id,
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            trait_name="Overall Persona Fit",
            trait_direction="matches persona style and values",
            model=self.judge_model,
            temperature=0.3,
            save_raw_log=False  # We'll save our own logs
        )

        # Determine winner (accounting for order randomization)
        winner_from_judge = judge_result.get("winner", "tie")

        if winner_from_judge == "A":
            actual_winner = "hybrid" if hybrid_is_a else "prompt"
        elif winner_from_judge == "B":
            actual_winner = "prompt" if hybrid_is_a else "hybrid"
        else:
            actual_winner = "tie"

        return {
            "persona_id": persona_id,
            "prompt": prompt,
            "response_hybrid": response_hybrid,
            "response_prompt": response_prompt,
            "winner": actual_winner,
            "confidence": judge_result.get("confidence", 0),
            "explanation": judge_result.get("explanation", ""),
            "hybrid_is_a": hybrid_is_a,
            "judge_winner": winner_from_judge
        }

    def run_validation(
        self,
        personas: List[str],
        prompts: List[str],
        resume_from: Optional[str] = None
    ):
        """
        Run large-scale validation experiment.

        Args:
            personas: List of persona IDs
            prompts: List of evaluation prompts
            resume_from: Path to checkpoint file to resume from
        """
        total_comparisons = len(personas) * len(prompts)

        print(f"\n{'='*80}")
        print(f"EXPERIMENT CONFIGURATION")
        print(f"{'='*80}")
        print(f"Personas: {len(personas)}")
        print(f"Prompts: {len(prompts)}")
        print(f"Total comparisons: {total_comparisons}")
        print(f"Judge model: {self.judge_model}")
        print(f"{'='*80}\n")

        # Load checkpoint if resuming
        completed = set()
        results = []

        if resume_from and Path(resume_from).exists():
            print(f"Resuming from checkpoint: {resume_from}")
            with open(resume_from) as f:
                checkpoint = json.load(f)
                results = checkpoint.get("results", [])
                completed = {(r["persona_id"], r["prompt"]) for r in results}
            print(f"✓ Loaded {len(completed)} completed comparisons\n")

        # Run comparisons
        with tqdm(total=total_comparisons, desc="Validation", initial=len(completed)) as pbar:
            for persona_id in personas:
                # Load optimized weights
                try:
                    weights = self._load_optimized_weights(persona_id)
                except FileNotFoundError as e:
                    print(f"\n⚠️  Skipping {persona_id}: {e}")
                    pbar.update(len(prompts))
                    continue

                for prompt in prompts:
                    # Skip if already completed
                    if (persona_id, prompt) in completed:
                        pbar.update(1)
                        continue

                    try:
                        # Generate responses
                        response_hybrid = self._generate_hybrid(persona_id, prompt, weights)
                        response_prompt = self._generate_prompt_only(persona_id, prompt)

                        # Evaluate
                        eval_result = self._evaluate_pair(
                            persona_id=persona_id,
                            prompt=prompt,
                            response_hybrid=response_hybrid,
                            response_prompt=response_prompt,
                            randomize_order=True
                        )

                        results.append(eval_result)

                        # Save checkpoint every 10 comparisons
                        if len(results) % 10 == 0:
                            self._save_checkpoint(results)

                    except Exception as e:
                        print(f"\n⚠️  Error for {persona_id} / {prompt[:50]}: {e}")
                        # Continue with next comparison

                    pbar.update(1)

        # Final save
        self._save_checkpoint(results)

        # Analyze and save summary
        summary = self._analyze_results(results)
        self._save_summary(summary)

        print(f"\n{'='*80}")
        print(f"VALIDATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total comparisons: {len(results)}")
        print(f"Results saved to: {self.checkpoint_dir}")
        print(f"{'='*80}\n")

        return results, summary

    def _save_checkpoint(self, results: List[Dict]):
        """Save checkpoint with all results."""
        checkpoint_file = self.checkpoint_dir / "comparison_results.json"

        checkpoint = {
            "date": datetime.now().isoformat(),
            "config": {
                "model": self.model_name,
                "layer": self.layer,
                "alpha": self.alpha,
                "judge_model": self.judge_model
            },
            "total_comparisons": len(results),
            "results": results
        }

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"  ✓ Checkpoint saved: {checkpoint_file}")

    def _analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze results and compute statistics."""
        # Count wins
        wins_hybrid = sum(1 for r in results if r["winner"] == "hybrid")
        wins_prompt = sum(1 for r in results if r["winner"] == "prompt")
        ties = sum(1 for r in results if r["winner"] == "tie")
        total = len(results)

        # Win rates
        win_rate_hybrid = wins_hybrid / total if total > 0 else 0
        win_rate_prompt = wins_prompt / total if total > 0 else 0
        tie_rate = ties / total if total > 0 else 0

        # Binomial test (H0: win_rate = 0.5)
        # Only count wins (exclude ties)
        decisive_comparisons = wins_hybrid + wins_prompt
        if decisive_comparisons > 0:
            # Use binomtest for newer scipy versions
            try:
                result = stats.binomtest(
                    wins_hybrid,
                    n=decisive_comparisons,
                    p=0.5,
                    alternative='two-sided'
                )
                binom_test = result.pvalue
            except AttributeError:
                # Fallback for older scipy versions
                binom_test = stats.binom_test(
                    wins_hybrid,
                    n=decisive_comparisons,
                    p=0.5,
                    alternative='two-sided'
                )
        else:
            binom_test = 1.0

        # Cohen's h (effect size for proportions)
        # h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
        if decisive_comparisons > 0:
            p1 = wins_hybrid / decisive_comparisons
            p2 = wins_prompt / decisive_comparisons
            cohens_h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
        else:
            cohens_h = 0.0

        # Wilson score 95% confidence interval for win rate
        if total > 0:
            ci_low, ci_high = self._wilson_score_interval(wins_hybrid, total)
        else:
            ci_low, ci_high = 0.0, 0.0

        # Average confidence scores
        avg_confidence = np.mean([r["confidence"] for r in results]) if results else 0

        # Per-persona breakdown
        persona_stats = {}
        for r in results:
            pid = r["persona_id"]
            if pid not in persona_stats:
                persona_stats[pid] = {"hybrid": 0, "prompt": 0, "tie": 0}
            persona_stats[pid][r["winner"]] += 1

        summary = {
            "date": datetime.now().isoformat(),
            "total_comparisons": total,
            "wins": {
                "hybrid": wins_hybrid,
                "prompt": wins_prompt,
                "ties": ties
            },
            "win_rates": {
                "hybrid": win_rate_hybrid,
                "prompt": win_rate_prompt,
                "tie": tie_rate
            },
            "statistical_tests": {
                "binomial_test_p_value": float(binom_test),
                "significant_at_0.05": binom_test < 0.05,
                "cohens_h": float(cohens_h),
                "effect_size_interpretation": self._interpret_cohens_h(cohens_h)
            },
            "confidence_interval_95": {
                "hybrid_win_rate_lower": float(ci_low),
                "hybrid_win_rate_upper": float(ci_high)
            },
            "average_judge_confidence": float(avg_confidence),
            "per_persona": persona_stats
        }

        return summary

    def _wilson_score_interval(self, wins: int, total: int, alpha: float = 0.05) -> Tuple[float, float]:
        """
        Calculate Wilson score confidence interval.

        More accurate than normal approximation for proportions.
        """
        if total == 0:
            return 0.0, 0.0

        p = wins / total
        z = stats.norm.ppf(1 - alpha / 2)  # 1.96 for 95% CI

        denominator = 1 + z**2 / total
        centre = (p + z**2 / (2 * total)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

        return centre - margin, centre + margin

    def _interpret_cohens_h(self, h: float) -> str:
        """Interpret Cohen's h effect size."""
        abs_h = abs(h)
        if abs_h < 0.2:
            return "negligible"
        elif abs_h < 0.5:
            return "small"
        elif abs_h < 0.8:
            return "medium"
        else:
            return "large"

    def _save_summary(self, summary: Dict):
        """Save analysis summary."""
        summary_file = self.checkpoint_dir / "summary.json"

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Also print to console
        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Total comparisons: {summary['total_comparisons']}")
        print(f"\nWin counts:")
        print(f"  Hybrid:       {summary['wins']['hybrid']:4d} ({summary['win_rates']['hybrid']*100:.1f}%)")
        print(f"  Prompt-only:  {summary['wins']['prompt']:4d} ({summary['win_rates']['prompt']*100:.1f}%)")
        print(f"  Ties:         {summary['wins']['ties']:4d} ({summary['win_rates']['tie']*100:.1f}%)")
        print(f"\nStatistical tests:")
        print(f"  Binomial test p-value: {summary['statistical_tests']['binomial_test_p_value']:.4f}")
        print(f"  Significant (p<0.05):  {summary['statistical_tests']['significant_at_0.05']}")
        print(f"  Cohen's h:             {summary['statistical_tests']['cohens_h']:.3f}")
        print(f"  Effect size:           {summary['statistical_tests']['effect_size_interpretation']}")
        print(f"\n95% CI for Hybrid win rate:")
        print(f"  [{summary['confidence_interval_95']['hybrid_win_rate_lower']:.3f}, {summary['confidence_interval_95']['hybrid_win_rate_upper']:.3f}]")
        print(f"\nAverage judge confidence: {summary['average_judge_confidence']:.2f}/5")
        print(f"{'='*80}\n")


def load_test_prompts() -> List[str]:
    """Load v2 prompts (test set - unused in optimization)."""
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

    print(f"✓ Loaded {len(all_prompts)} test prompts from v2")
    return all_prompts


def load_optimized_personas() -> List[str]:
    """Load list of personas with optimized weights."""
    # Load from analysis summary
    summary_file = Path("optimization_results_26personas/analysis_summary.json")

    if not summary_file.exists():
        raise FileNotFoundError(f"Analysis summary not found: {summary_file}")

    with open(summary_file) as f:
        data = json.load(f)

    personas = [p["persona_id"] for p in data["personas"]]
    print(f"✓ Loaded {len(personas)} optimized personas")
    return personas


def main():
    parser = argparse.ArgumentParser(
        description="Large-scale Hybrid validation experiment"
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID (default: 0)"
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=20,
        help="Steering layer (default: 20)"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Steering strength (default: 2.0)"
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="Judge model (default: gpt-4o)"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint file"
    )

    parser.add_argument(
        "--limit-personas",
        type=int,
        default=None,
        help="Limit number of personas (for testing)"
    )

    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=None,
        help="Limit number of prompts (for testing)"
    )

    args = parser.parse_args()

    # Set device
    device = f"cuda:{args.gpu_id}"

    # Load data
    personas = load_optimized_personas()
    prompts = load_test_prompts()

    # Apply limits if specified (for testing)
    if args.limit_personas:
        personas = personas[:args.limit_personas]
        print(f"⚠️  Limited to {len(personas)} personas for testing")

    if args.limit_prompts:
        prompts = prompts[:args.limit_prompts]
        print(f"⚠️  Limited to {len(prompts)} prompts for testing")

    # Initialize validator
    validator = HybridValidator(
        layer=args.layer,
        alpha=args.alpha,
        device=device,
        judge_model=args.judge_model
    )

    # Run validation
    results, summary = validator.run_validation(
        personas=personas,
        prompts=prompts,
        resume_from=args.resume
    )

    print("\n✅ Validation experiment complete!")
    print(f"Results: results/hybrid_validation_large_scale/")


if __name__ == "__main__":
    main()
