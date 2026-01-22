"""
Human Evaluation Framework.

Generates randomized comparison data for human evaluators.
"""

import json
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import torch

from .utils import (
    load_optimization_results,
    build_combined_steering_vector,
    load_persona_profile,
    EvaluationConfig
)


class HumanEvalGenerator:
    """Generates data for human evaluation studies."""

    def __init__(
        self,
        config: EvaluationConfig,
        steerer,
        optimization_dir: str = "persona-opt",
        vectors_dir: str = "data/steering_vectors_v2"
    ):
        """
        Initialize generator.

        Args:
            config: Evaluation configuration
            steerer: Llama3ActivationSteerer instance
            optimization_dir: Directory with optimization results
            vectors_dir: Directory with steering vectors
        """
        self.config = config
        self.steerer = steerer
        self.optimization_dir = optimization_dir
        self.vectors_dir = vectors_dir

        # Load optimization results
        self.opt_results = load_optimization_results(config.persona_id, optimization_dir)
        self.weights = self.opt_results['weights']
        self.alpha = self.opt_results.get('alpha', 2.0)
        self.optimized_layer = self.opt_results.get('layer', config.layer)

        # Load persona profile
        self.persona_profile = load_persona_profile(config.persona_id)

    def generate_evaluation_set(
        self,
        prompts: List[str],
        num_samples: int = 20,
        include_persona_sample: bool = True,
        output_dir: Optional[str] = None,
        seed: int = 42
    ) -> Dict:
        """
        Generate human evaluation dataset.

        Args:
            prompts: List of evaluation prompts
            num_samples: Number of samples to generate
            include_persona_sample: Whether to include real persona sample
            output_dir: Output directory for results
            seed: Random seed for reproducibility

        Returns:
            Results dictionary with evaluation data
        """
        random.seed(seed)
        print(f"[HumanEval] Generating evaluation set for {self.config.persona_id}")
        print(f"[HumanEval] Samples: {num_samples}")

        # Sample prompts
        selected_prompts = random.sample(prompts, min(num_samples, len(prompts)))

        # Load steering vectors
        trait_vectors = {}
        traits = ["R1", "R2", "R3", "R4", "R5"]
        for trait in traits:
            vector_file = Path(self.vectors_dir) / trait / f"layer{self.optimized_layer}_svd.pt"
            vector_data = torch.load(vector_file, map_location='cpu')
            # Handle different formats - check tensor first
            if isinstance(vector_data, torch.Tensor):
                trait_vectors[trait] = vector_data
            elif isinstance(vector_data, dict) and 'vector' in vector_data:
                trait_vectors[trait] = vector_data['vector']
            else:
                raise ValueError(f"Unexpected vector format in {vector_file}")

        # Build combined steering vector
        steering_vector = build_combined_steering_vector(self.weights, trait_vectors)

        # Generate responses
        print(f"[HumanEval] Generating baseline responses...")
        baseline_responses = self.steerer.batch_generate(selected_prompts)

        print(f"[HumanEval] Generating steered responses...")
        steered_responses = self.steerer.batch_generate(
            selected_prompts,
            steering_vector=steering_vector,
            layer=self.optimized_layer,
            alpha=self.alpha
        )

        # Get persona samples if available
        persona_samples = []
        if include_persona_sample and 'sample_responses' in self.persona_profile:
            persona_samples = self.persona_profile['sample_responses'][:num_samples]

        # Create evaluation items with randomization
        evaluation_items = []
        for i, prompt in enumerate(selected_prompts):
            item = {
                'item_id': i + 1,
                'prompt': prompt,
                'response_A': '',
                'response_B': '',
                'response_C': '' if include_persona_sample and i < len(persona_samples) else None,
                'true_labels': {},
                'randomization_seed': random.randint(0, 999999)
            }

            # Randomize order of baseline and steered
            responses = [
                ('baseline', baseline_responses[i]),
                ('steered', steered_responses[i])
            ]

            if include_persona_sample and i < len(persona_samples):
                responses.append(('persona', persona_samples[i]))

            # Shuffle responses
            random.shuffle(responses)

            # Assign to A, B, C
            item['response_A'] = responses[0][1]
            item['true_labels']['A'] = responses[0][0]

            item['response_B'] = responses[1][1]
            item['true_labels']['B'] = responses[1][0]

            if len(responses) > 2:
                item['response_C'] = responses[2][1]
                item['true_labels']['C'] = responses[2][0]

            evaluation_items.append(item)

        results = {
            'evaluation_type': 'human_evaluation',
            'persona_id': self.config.persona_id,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'num_samples': num_samples,
                'include_persona_sample': include_persona_sample,
                'seed': seed
            },
            'persona_profile': {
                'persona_id': self.persona_profile['persona_id'],
                'traits': self.persona_profile.get('traits', {}),
                'description': self.persona_profile.get('description', '')
            },
            'evaluation_items': evaluation_items,
            'instructions': self._generate_instructions()
        }

        # Save outputs
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save full JSON
            json_file = output_path / 'evaluation_data.json'
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2)

            # Save CSV for human evaluators (without true labels)
            csv_file = output_path / 'human_evaluation.csv'
            self._save_csv(evaluation_items, csv_file, include_labels=False)

            # Save answer key CSV
            key_file = output_path / 'answer_key.csv'
            self._save_csv(evaluation_items, key_file, include_labels=True)

            # Save instructions
            instructions_file = output_path / 'instructions.md'
            with open(instructions_file, 'w') as f:
                f.write(results['instructions'])

            print(f"\n[HumanEval] Files saved:")
            print(f"  - {json_file}")
            print(f"  - {csv_file}")
            print(f"  - {key_file}")
            print(f"  - {instructions_file}")

        return results

    def _save_csv(self, items: List[Dict], filepath: Path, include_labels: bool):
        """Save evaluation items to CSV."""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            # Determine if 3-way comparison
            has_c = items[0]['response_C'] is not None

            fieldnames = ['item_id', 'prompt', 'response_A', 'response_B']
            if has_c:
                fieldnames.append('response_C')

            if include_labels:
                fieldnames.extend(['label_A', 'label_B'])
                if has_c:
                    fieldnames.append('label_C')
            else:
                fieldnames.extend(['rating_A', 'rating_B'])
                if has_c:
                    fieldnames.append('rating_C')
                fieldnames.extend(['best_response', 'notes'])

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for item in items:
                row = {
                    'item_id': item['item_id'],
                    'prompt': item['prompt'],
                    'response_A': item['response_A'],
                    'response_B': item['response_B']
                }

                if has_c:
                    row['response_C'] = item['response_C']

                if include_labels:
                    row['label_A'] = item['true_labels']['A']
                    row['label_B'] = item['true_labels']['B']
                    if has_c:
                        row['label_C'] = item['true_labels']['C']
                else:
                    row['rating_A'] = ''
                    row['rating_B'] = ''
                    if has_c:
                        row['rating_C'] = ''
                    row['best_response'] = ''
                    row['notes'] = ''

                writer.writerow(row)

    def _generate_instructions(self) -> str:
        """Generate human evaluation instructions."""
        return f"""# Human Evaluation Instructions

## Task

You will evaluate different responses to conversational prompts. Your goal is to determine which response **best matches the persona** described below.

## Persona Description

**Persona ID**: {self.persona_profile['persona_id']}

**Traits**:
{self._format_traits()}

**Communication Style**: {self.persona_profile.get('description', 'See traits above')}

## Evaluation Process

For each item:

1. **Read the prompt** carefully
2. **Read all responses** (A, B, and possibly C)
3. **Rate each response** on how well it matches the persona (1-5 scale):
   - 1 = Does not match persona at all
   - 2 = Slightly matches persona
   - 3 = Moderately matches persona
   - 4 = Closely matches persona
   - 5 = Perfectly matches persona

4. **Select the best response** (A, B, or C)
5. **Add notes** if needed to explain your reasoning

## Important Notes

- Focus on **persona fit**, not general quality
- Consider **tone, style, word choice, and perspective**
- The responses are randomized - don't assume any pattern
- Trust your intuition about what "sounds like" this persona
- You can leave notes for items where it's unclear

## Time Estimate

- Approximately 2-3 minutes per item
- Total time: {len(self.persona_profile.get('sample_responses', []))} items × 2.5 min ≈ {len(self.persona_profile.get('sample_responses', [])) * 2.5:.0f} minutes

Thank you for your time and careful evaluation!
"""

    def _format_traits(self) -> str:
        """Format persona traits for instructions."""
        traits = self.persona_profile.get('traits', {})
        if not traits:
            return "  (No specific traits listed)"

        lines = []
        for key, value in traits.items():
            lines.append(f"  - **{key}**: {value}")
        return "\n".join(lines)
