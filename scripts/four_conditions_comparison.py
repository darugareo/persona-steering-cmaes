#!/usr/bin/env python3
"""
4条件比較実験: Base, Steering, Prompt, Hybrid
context付きプロンプトで正しく評価
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys
import random
import os
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from persona_opt.internal_steering_l3 import Llama3ActivationSteerer

class FourConditionsComparator:
    def __init__(
        self,
        persona_id: str,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        layer: int = 20,
        alpha: float = 2.0,
        device: str = "cuda:0"
    ):
        self.persona_id = persona_id
        self.layer = layer
        self.alpha = alpha
        self.device = device

        # Load persona data
        profile_path = Path(f"personas_cc/{persona_id}/profile.json")

        # Use selected high-quality turns if available
        test_path_selected = Path(f"personas_cc/{persona_id}/test_turns_selected.json")
        test_path_default = Path(f"personas_cc/{persona_id}/test_turns.json")

        if test_path_selected.exists():
            test_path = test_path_selected
            print(f"Using selected high-quality turns")
        else:
            test_path = test_path_default
            print(f"Using default test turns")

        with open(profile_path) as f:
            self.profile = json.load(f)

        with open(test_path) as f:
            test_data = json.load(f)
            self.test_turns = test_data["turns"][:10]

        # Load model
        print(f"Loading model: {model_name}")
        self.steerer = Llama3ActivationSteerer(
            model_name=model_name,
            target_layer=layer,
            device=device
        )

        # Load optimized weights (Style fitness)
        weight_file = Path(f"results/fitness_comparison/optimization_logs/{persona_id}_style.json")
        with open(weight_file) as f:
            opt_result = json.load(f)
            self.optimized_weights = opt_result["best_weights"]

        print(f"Loaded optimized weights: {self.optimized_weights}")

        # Load trait vectors
        self.trait_vectors = self._load_trait_vectors()

        # OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def _load_trait_vectors(self) -> Dict[str, torch.Tensor]:
        """Load inverted trait vectors"""
        trait_dir = Path("data/steering_vectors_v2_inverted")
        vectors = {}

        for trait in ["R1", "R2", "R3", "R4", "R5"]:
            vector_path = trait_dir / trait / "layer20_svd.pt"
            vector = torch.load(vector_path, map_location='cpu', weights_only=False)
            vectors[trait] = vector.to(self.device)

        return vectors

    def generate_base(self, context: str, input_text: str) -> str:
        """Condition 1: Base (no steering, no persona prompt)"""
        prompt = f"""Continue this conversation naturally.

Conversation so far:
{context}

Partner: {input_text}

You:"""

        return self.steerer.generate(
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7
        )

    def generate_steering(self, context: str, input_text: str) -> str:
        """Condition 2: Steering-only (steering with optimized weights, no persona prompt)"""
        prompt = f"""Continue this conversation naturally.

Conversation so far:
{context}

Partner: {input_text}

You:"""

        # Apply steering
        scaled_weights = {k: v * self.alpha for k, v in self.optimized_weights.items()}
        self.steerer.register_hooks(
            multi_trait_vectors=self.trait_vectors,
            trait_weights=scaled_weights
        )

        response = self.steerer.generate(
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7
        )

        self.steerer.remove_hooks()
        return response

    def generate_prompt(self, context: str, input_text: str) -> str:
        """Condition 3: Prompt-only (no steering, with persona prompt)"""
        examples = "\n".join([f"- {ex}" for ex in self.profile.get("example_utterances", [])[:5]])

        prompt = f"""You are {self.profile['speaker_role']} in a {self.profile['relationship']} relationship.

Your communication style examples:
{examples}

Conversation so far:
{context}

Partner: {input_text}

Respond naturally as {self.profile['speaker_role']}:"""

        return self.steerer.generate(
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7
        )

    def generate_hybrid(self, context: str, input_text: str) -> str:
        """Condition 4: Hybrid (steering + persona prompt)"""
        examples = "\n".join([f"- {ex}" for ex in self.profile.get("example_utterances", [])[:5]])

        prompt = f"""You are {self.profile['speaker_role']} in a {self.profile['relationship']} relationship.

Your communication style examples:
{examples}

Conversation so far:
{context}

Partner: {input_text}

Respond naturally as {self.profile['speaker_role']}:"""

        # Apply steering
        scaled_weights = {k: v * self.alpha for k, v in self.optimized_weights.items()}
        self.steerer.register_hooks(
            multi_trait_vectors=self.trait_vectors,
            trait_weights=scaled_weights
        )

        response = self.steerer.generate(
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7
        )

        self.steerer.remove_hooks()
        return response

    def judge_comparison(
        self,
        response_a: str,
        response_b: str,
        ground_truth: str,
        context: str,
        turn_id: int
    ) -> str:
        """GPT-4o Judge: which response is more similar to ground truth?"""
        try:
            # Randomize order to avoid position bias
            if random.random() < 0.5:
                actual_a, actual_b = response_a, response_b
                order = "original"
            else:
                actual_a, actual_b = response_b, response_a
                order = "swapped"

            prompt = f"""You are evaluating which response better matches the target persona's actual response.

## Context
{context}

## Target Persona's Actual Response (Ground Truth)
"{ground_truth}"

## Response A
"{actual_a}"

## Response B
"{actual_b}"

Which response is MORE SIMILAR to the ground truth in terms of:
- Writing style and tone
- Word choice and expressions
- Personality and attitude

Output only "A" or "B" (no ties allowed)."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5
            )

            result = response.choices[0].message.content.strip().upper()

            # Map back to original order
            if order == "swapped":
                if result == "A":
                    result = "B"
                elif result == "B":
                    result = "A"

            return result

        except Exception as e:
            print(f"Warning: Judge comparison failed for turn {turn_id}: {e}")
            return "ERROR"

    def run_experiment(self) -> Dict:
        """Run full 4-condition experiment"""
        print(f"\n{'='*80}")
        print(f"Running 4-condition comparison: {self.persona_id}")
        print(f"{'='*80}\n")

        generations = []

        # Generate for all 4 conditions
        for i, turn in enumerate(self.test_turns):
            context = turn["context"]
            input_text = turn["input"]
            ground_truth = turn["ground_truth"]

            print(f"Turn {i+1}/10: Generating...")

            base = self.generate_base(context, input_text)
            steering = self.generate_steering(context, input_text)
            prompt = self.generate_prompt(context, input_text)
            hybrid = self.generate_hybrid(context, input_text)

            generations.append({
                "turn_id": i + 1,
                "context": context,
                "input": input_text,
                "ground_truth": ground_truth,
                "base": base,
                "steering": steering,
                "prompt": prompt,
                "hybrid": hybrid
            })

        print(f"\n{'='*80}")
        print(f"Running Judge comparisons...")
        print(f"{'='*80}\n")

        # Pairwise comparisons
        comparisons = {
            "steering_vs_base": [],
            "prompt_vs_steering": [],
            "hybrid_vs_prompt": []
        }

        for gen in generations:
            turn_id = gen["turn_id"]
            context = gen["context"]
            ground_truth = gen["ground_truth"]

            print(f"Turn {turn_id}/10: Judging...")

            # Steering vs Base
            result_1 = self.judge_comparison(
                gen["steering"], gen["base"],
                ground_truth, context, turn_id
            )
            comparisons["steering_vs_base"].append({
                "turn_id": turn_id,
                "winner": "steering" if result_1 == "A" else "base" if result_1 == "B" else "error"
            })

            # Prompt vs Steering
            result_2 = self.judge_comparison(
                gen["prompt"], gen["steering"],
                ground_truth, context, turn_id
            )
            comparisons["prompt_vs_steering"].append({
                "turn_id": turn_id,
                "winner": "prompt" if result_2 == "A" else "steering" if result_2 == "B" else "error"
            })

            # Hybrid vs Prompt
            result_3 = self.judge_comparison(
                gen["hybrid"], gen["prompt"],
                ground_truth, context, turn_id
            )
            comparisons["hybrid_vs_prompt"].append({
                "turn_id": turn_id,
                "winner": "hybrid" if result_3 == "A" else "prompt" if result_3 == "B" else "error"
            })

        return {
            "persona_id": self.persona_id,
            "generations": generations,
            "comparisons": comparisons
        }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona_id", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"

    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    print(f"✅ OPENAI_API_KEY is set")

    # Run experiment
    comparator = FourConditionsComparator(
        persona_id=args.persona_id,
        device=device
    )

    results = comparator.run_experiment()

    # Save results
    output_dir = Path("results/four_conditions")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save generations
    gen_dir = output_dir / "generations"
    gen_dir.mkdir(exist_ok=True)

    gen_path = gen_dir / f"{args.persona_id}.json"
    with open(gen_path, "w") as f:
        json.dump({
            "persona_id": results["persona_id"],
            "generations": results["generations"]
        }, f, indent=2)

    # Save comparisons
    comp_dir = output_dir / "comparisons"
    comp_dir.mkdir(exist_ok=True)

    for comp_name, comp_data in results["comparisons"].items():
        comp_path = comp_dir / f"{args.persona_id}_{comp_name}.json"
        with open(comp_path, "w") as f:
            json.dump({
                "persona_id": results["persona_id"],
                "comparison": comp_name,
                "results": comp_data
            }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Experiment complete!")
    print(f"  Generations: {gen_path}")
    print(f"  Comparisons: {comp_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
