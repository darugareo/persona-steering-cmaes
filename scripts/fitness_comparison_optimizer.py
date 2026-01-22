#!/usr/bin/env python3
"""
Fitness Function Comparison Optimizer
小規模テスト: 2ペルソナ × 4 Fitness関数
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import cma
from transformers import AutoTokenizer, AutoModel
from bert_score import score as bert_score_fn
import openai
import os

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.internal_steering_l3 import Llama3ActivationSteerer

class FitnessComparator:
    def __init__(
        self,
        persona_id: str,
        fitness_type: str,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        layer: int = 20,
        alpha: float = 2.0,
        device: str = "cuda:0"
    ):
        self.persona_id = persona_id
        self.fitness_type = fitness_type
        self.layer = layer
        self.alpha = alpha
        self.device = device
        
        # Load persona data
        profile_path = Path(f"personas_cc/{persona_id}/profile.json")
        train_path = Path(f"personas_cc/{persona_id}/train_turns.json")

        with open(profile_path) as f:
            self.profile = json.load(f)

        with open(train_path) as f:
            train_data = json.load(f)
            self.train_turns = train_data["turns"][:10]  # Use first 10 turns for optimization
        
        # Load model
        print(f"Loading model: {model_name}")
        self.steerer = Llama3ActivationSteerer(
            model_name=model_name,
            target_layer=layer,
            device=device
        )
        self.alpha = alpha
        
        # Load inverted trait vectors
        self.trait_vectors = self._load_trait_vectors()
        
        # Setup fitness function
        if fitness_type == "bertscore":
            self.fitness_fn = self.fitness_bertscore
        elif fitness_type == "style":
            self.fitness_fn = self.fitness_style
        elif fitness_type == "judge":
            self.fitness_fn = self.fitness_judge
        elif fitness_type == "combined":
            self.fitness_fn = self.fitness_combined
        else:
            raise ValueError(f"Unknown fitness type: {fitness_type}")
        
        print(f"Initialized: {persona_id} with {fitness_type} fitness")
    
    def _load_trait_vectors(self) -> Dict[str, torch.Tensor]:
        """Load inverted trait vectors"""
        trait_dir = Path("data/steering_vectors_v2_inverted")
        vectors = {}
        
        for trait in ["R1", "R2", "R3", "R4", "R5"]:
            vector_path = trait_dir / trait / "layer20_svd.pt"
            vector = torch.load(vector_path, map_location='cpu', weights_only=False)
            vectors[trait] = vector.to(self.device)
        
        return vectors
    
    def generate_with_steering(self, prompt: str, trait_weights: Dict[str, float]) -> str:
        """Generate text with steering"""
        # Apply alpha scaling to trait weights
        scaled_weights = {k: v * self.alpha for k, v in trait_weights.items()}

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
    
    def fitness_bertscore(self, generated: str, ground_truth: str, context: str = "") -> float:
        """Fitness A: BERTScore"""
        try:
            P, R, F1 = bert_score_fn([generated], [ground_truth], lang="en", verbose=False)
            return F1.item()
        except Exception as e:
            print(f"BERTScore error: {e}")
            return 0.0
    
    def fitness_style(self, generated: str, ground_truth: str, context: str = "") -> float:
        """Fitness B: Style Similarity"""
        def extract_features(text):
            words = text.lower().split()
            if len(words) == 0:
                return {k: 0.0 for k in ["first_person", "question", "exclamation", "avg_len", "casual"]}
            
            return {
                "first_person": sum(1 for w in words if w in ['i', 'my', 'me', "i'm", "i've"]) / len(words),
                "question": text.count('?') / max(len(text), 1),
                "exclamation": text.count('!') / max(len(text), 1),
                "avg_len": sum(len(w) for w in words) / len(words),
                "casual": sum(1 for w in words if w in ['gonna', 'wanna', 'yeah', 'hey', 'wow', 'like']) / len(words),
            }
        
        gen_feat = extract_features(generated)
        gt_feat = extract_features(ground_truth)
        
        # Cosine similarity
        a = np.array(list(gen_feat.values()))
        b = np.array(list(gt_feat.values()))
        
        similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return max(0.0, min(1.0, similarity))
    
    def fitness_judge(self, generated: str, ground_truth: str, context: str = "") -> float:
        """Fitness C: GPT-4o Judge"""
        try:
            # Check if API key is set
            import os
            if not os.getenv('OPENAI_API_KEY'):
                # Silently return 0.5 if no API key (avoid spam)
                return 0.5

            # Prepare persona info
            role = self.profile.get('speaker_role', 'Unknown')
            relationship = self.profile.get('relationship', 'Unknown')
            examples = self.profile.get('example_utterances', [])[:3]

            prompt = f"""You are evaluating persona consistency in dialogue generation.

Persona Profile:
- Role: {role}
- Relationship: {relationship}
- Example utterances:
{chr(10).join(f"  {i+1}. {ex}" for i, ex in enumerate(examples))}

Ground Truth Utterance: "{ground_truth}"
Generated Utterance: "{generated}"

Rate how well the generated utterance matches the persona's style, tone, and characteristics.
Consider:
- Speaking style consistency
- Tone and formality level
- Word choice and vocabulary
- Typical expressions

Provide ONLY a single number between 0.0 and 1.0:
- 1.0: Perfect match
- 0.5: Partially matches
- 0.0: Completely different

Score:"""

            # Use new OpenAI API v1.0+
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )

            score_text = response.choices[0].message.content.strip()
            # Extract first number found
            import re
            match = re.search(r'(\d+\.?\d*)', score_text)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            else:
                print(f"Warning: Could not parse GPT-4o response: {score_text}")
                return 0.5

        except Exception as e:
            # Only print error once per run to avoid spam
            if not hasattr(self, '_judge_error_printed'):
                print(f"GPT-4o Judge error: {e}")
                self._judge_error_printed = True
            return 0.5
    
    def fitness_combined(self, generated: str, ground_truth: str, context: str = "") -> float:
        """Fitness D: Combined"""
        bert = self.fitness_bertscore(generated, ground_truth, context)
        style = self.fitness_style(generated, ground_truth, context)
        judge = self.fitness_judge(generated, ground_truth, context)
        
        return 0.4 * bert + 0.3 * style + 0.3 * judge
    
    def objective_function(self, weights: np.ndarray) -> float:
        """CMA-ES objective function (to minimize, so return negative fitness)"""
        trait_weights = {
            "R1": float(weights[0]),
            "R2": float(weights[1]),
            "R3": float(weights[2]),
            "R4": float(weights[3]),
            "R5": float(weights[4])
        }
        
        total_fitness = 0.0

        # Evaluate on train turns
        for turn in self.train_turns:
            # Build prompt with conversation context
            context = turn["context"]
            input_text = turn["input"]
            ground_truth = turn["ground_truth"]

            prompt = f"""Continue this conversation naturally.

Conversation so far:
{context}

Partner: {input_text}

You:"""

            # Generate
            generated = self.generate_with_steering(prompt, trait_weights)

            # Calculate fitness
            fitness = self.fitness_fn(generated, ground_truth, context)
            total_fitness += fitness

        avg_fitness = total_fitness / len(self.train_turns)
        return -avg_fitness  # Negative for minimization
    
    def optimize(self, max_generations: int = 10) -> Dict:
        """Run CMA-ES optimization"""
        print(f"\nStarting CMA-ES optimization...")
        print(f"  Persona: {self.persona_id}")
        print(f"  Fitness: {self.fitness_type}")
        print(f"  Max generations: {max_generations}")
        
        # CMA-ES setup
        initial_mean = np.zeros(5)
        initial_sigma = 1.0
        
        es = cma.CMAEvolutionStrategy(initial_mean, initial_sigma, {
            'popsize': 8,
            'maxiter': max_generations,
            'verb_disp': 1,
            'verb_log': 0
        })
        
        generation = 0
        best_fitness = float('inf')
        best_weights = None
        
        while not es.stop() and generation < max_generations:
            solutions = es.ask()
            fitness_values = [self.objective_function(x) for x in solutions]
            es.tell(solutions, fitness_values)
            
            # Track best
            current_best_idx = np.argmin(fitness_values)
            if fitness_values[current_best_idx] < best_fitness:
                best_fitness = fitness_values[current_best_idx]
                best_weights = solutions[current_best_idx]
            
            generation += 1
            print(f"  Gen {generation}: best_fitness={-best_fitness:.4f}")
        
        result = {
            "persona_id": self.persona_id,
            "fitness_type": self.fitness_type,
            "best_weights": {
                "R1": float(best_weights[0]),
                "R2": float(best_weights[1]),
                "R3": float(best_weights[2]),
                "R4": float(best_weights[3]),
                "R5": float(best_weights[4])
            },
            "best_fitness": float(-best_fitness),
            "generations": generation
        }
        
        return result

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona_id", required=True)
    parser.add_argument("--fitness_type", required=True, choices=["bertscore", "style", "judge", "combined"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_generations", type=int, default=10)
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu_id}"
    
    # Run optimization
    comparator = FitnessComparator(
        persona_id=args.persona_id,
        fitness_type=args.fitness_type,
        device=device
    )
    
    result = comparator.optimize(max_generations=args.max_generations)
    
    # Save result
    output_dir = Path("results/fitness_comparison/optimization_logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{args.persona_id}_{args.fitness_type}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Optimization complete!")
    print(f"  Best fitness: {result['best_fitness']:.4f}")
    print(f"  Best weights: {result['best_weights']}")
    print(f"  Saved to: {output_path}")

if __name__ == "__main__":
    main()
