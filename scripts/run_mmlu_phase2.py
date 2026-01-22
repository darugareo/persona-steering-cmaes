"""
Phase 2 - MMLU Subset Evaluation
Evaluates Base, Prompt Persona, and Proposed methods on MMLU to assess capability impact.
"""

import json
import sys
import re
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from datasets import load_dataset
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.utils.config_loader import (
    load_experiment_config,
    set_seed,
)
from persona_opt.internal_steering_l3 import Llama3ActivationSteerer
from persona_opt.baselines import (
    BaseMethod,
    PromptPersonaMethod,
    ProposedMethod
)
from persona_opt.evaluation.utils import load_persona_profile


# MMLU Subset categories
STEM_SUBJECTS = [
    'abstract_algebra',
    'anatomy',
    'astronomy',
    'college_biology',
    'college_chemistry',
    'college_mathematics',
    'college_physics',
    'elementary_mathematics',
    'high_school_biology',
    'high_school_chemistry',
    'high_school_mathematics',
    'high_school_physics'
]

HUMANITIES_SUBJECTS = [
    'formal_logic',
    'high_school_european_history',
    'high_school_us_history',
    'high_school_world_history',
    'international_law',
    'jurisprudence',
    'logical_fallacies',
    'moral_disputes',
    'moral_scenarios',
    'philosophy',
    'prehistory',
    'professional_law',
    'world_religions'
]


class MMLUEvaluator:
    """MMLU evaluation for persona steering methods."""

    def __init__(
        self,
        persona_id: str,
        seed: int = 1,
        subset: bool = True,
        num_samples_per_subject: int = None,
        config_path: str = "config/experiment_config.yaml"
    ):
        self.persona_id = persona_id
        self.seed = seed
        self.subset = subset
        self.num_samples_per_subject = num_samples_per_subject

        # Load config and set seed
        self.config = load_experiment_config(config_path)
        set_seed(seed, deterministic=self.config.deterministic)

        # Load persona profile
        self.persona_profile = load_persona_profile(persona_id)

        # Initialize steerer
        print(f"\n[MMLU] Initializing steerer...")
        self.steerer = Llama3ActivationSteerer(
            model_name=self.config.model_name,
            target_layer=self.config.default_layer,
            device=self.config.device,
        )

        # Determine subjects
        if subset:
            # Use smaller subset for faster evaluation
            self.subjects = {
                'STEM': STEM_SUBJECTS[:4],  # 4 STEM subjects
                'Humanities': HUMANITIES_SUBJECTS[:4]  # 4 Humanities subjects
            }
        else:
            self.subjects = {
                'STEM': STEM_SUBJECTS,
                'Humanities': HUMANITIES_SUBJECTS
            }

        # Load dataset
        print(f"[MMLU] Loading dataset...")
        self.dataset = self._load_mmlu()

        self.results = {}

    def _load_mmlu(self):
        """Load MMLU dataset from HuggingFace."""
        all_data = {}

        for category, subjects in self.subjects.items():
            all_data[category] = {}

            for subject in subjects:
                try:
                    # Load subject dataset
                    dataset = load_dataset("cais/mmlu", subject, trust_remote_code=True)

                    # Use test split
                    data = dataset['test']

                    # Sample if needed
                    if self.num_samples_per_subject and self.num_samples_per_subject < len(data):
                        np.random.seed(self.seed)
                        indices = np.random.choice(len(data), self.num_samples_per_subject, replace=False)
                        data = data.select(indices)

                    all_data[category][subject] = data
                    print(f"  Loaded {subject}: {len(data)} samples")

                except Exception as e:
                    print(f"  Warning: Could not load {subject}: {e}")

        return all_data

    def _format_mmlu_prompt(self, question: str, choices: List[str]) -> str:
        """Format MMLU question as a prompt."""
        choice_labels = ['A', 'B', 'C', 'D']
        prompt = f"{question}\n\n"

        for label, choice in zip(choice_labels[:len(choices)], choices):
            prompt += f"{label}. {choice}\n"

        prompt += "\nAnswer with only the letter (A, B, C, or D):"
        return prompt

    def _extract_answer(self, response: str) -> str:
        """Extract answer choice from model response."""
        # Look for A, B, C, D in the response
        response = response.strip().upper()

        # Try to find first occurrence of A, B, C, or D
        match = re.search(r'\b([ABCD])\b', response)
        if match:
            return match.group(1)

        # Fallback: check first character
        if response and response[0] in 'ABCD':
            return response[0]

        return None

    def _evaluate_accuracy(self, response: str, correct_answer: int) -> bool:
        """Evaluate if response is correct."""
        predicted = self._extract_answer(response)
        if predicted is None:
            return False

        # Convert correct_answer (0-3) to letter (A-D)
        correct_letter = chr(65 + correct_answer)  # 65 = ord('A')

        return predicted == correct_letter

    def evaluate_method(
        self,
        method_name: str,
        method_instance
    ) -> Dict:
        """Evaluate a single method on MMLU."""

        print(f"\n{'='*80}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*80}")

        category_results = defaultdict(lambda: {'correct': 0, 'total': 0, 'details': []})
        all_correct = 0
        all_total = 0

        for category, subjects_dict in self.dataset.items():
            print(f"\n[{category}]")

            for subject, data in subjects_dict.items():
                subject_correct = 0

                for i, item in enumerate(data):
                    question = item['question']
                    choices = item['choices']
                    correct_answer = item['answer']

                    # Format prompt
                    prompt = self._format_mmlu_prompt(question, choices)

                    # Generate response
                    self.steerer.remove_hooks()
                    if method_name == "base":
                        response = self.steerer.generate(
                            prompt,
                            max_new_tokens=10,  # Short answer
                            temperature=0.0  # Greedy for accuracy
                        )
                    else:
                        response = method_instance.generate(
                            prompt,
                            max_new_tokens=10,
                            temperature=0.0
                        )

                    # Evaluate
                    is_correct = self._evaluate_accuracy(response, correct_answer)

                    if is_correct:
                        subject_correct += 1
                        all_correct += 1

                    all_total += 1

                    category_results[category]['details'].append({
                        'subject': subject,
                        'question': question,
                        'response': response,
                        'correct_answer': chr(65 + correct_answer),
                        'is_correct': is_correct
                    })

                subject_acc = subject_correct / len(data) if len(data) > 0 else 0
                category_results[category]['correct'] += subject_correct
                category_results[category]['total'] += len(data)

                print(f"  {subject}: {subject_acc:.3f} ({subject_correct}/{len(data)})")

            # Print category summary
            cat_acc = category_results[category]['correct'] / category_results[category]['total'] if category_results[category]['total'] > 0 else 0
            print(f"\n  [{category} Overall]: {cat_acc:.3f} ({category_results[category]['correct']}/{category_results[category]['total']})")

        # Overall accuracy
        overall_accuracy = all_correct / all_total if all_total > 0 else 0

        result = {
            'method': method_name,
            'overall_accuracy': overall_accuracy,
            'total_correct': all_correct,
            'total_questions': all_total,
            'category_results': dict(category_results),
            'seed': self.seed
        }

        print(f"\n[{method_name}] Overall MMLU Accuracy: {overall_accuracy:.3f}")

        return result

    def run_all_methods(self):
        """Run evaluation on all methods: base, prompt_persona, proposed."""

        # 1. Base Method
        print(f"\n[1/3] Base Method (No Steering)")
        base_method = BaseMethod(
            steerer=self.steerer,
            persona_id=self.persona_id
        )
        self.results['base'] = self.evaluate_method('base', base_method)

        # 2. Prompt Persona Method
        print(f"\n[2/3] Prompt Persona Method")
        prompt_method = PromptPersonaMethod(
            steerer=self.steerer,
            persona_id=self.persona_id,
            persona_profile=self.persona_profile
        )
        self.results['prompt_persona'] = self.evaluate_method('prompt_persona', prompt_method)

        # 3. Proposed Method
        print(f"\n[3/3] Proposed Method (SVD + CMA-ES)")
        proposed_method = ProposedMethod(
            steerer=self.steerer,
            persona_id=self.persona_id,
            layer=self.config.default_layer,
            alpha=self.config.default_alpha
        )
        self.results['proposed'] = self.evaluate_method('proposed', proposed_method)

        return self.results

    def save_results(self):
        """Save results to JSON and Markdown."""

        # Create output directory (persona-specific)
        output_dir = Path(f"reports/{self.persona_id}/phase2/mmlu")
        output_dir.mkdir(parents=True, exist_ok=True)

        subset_str = "subset_" if self.subset else ""
        json_path = output_dir / f"mmlu_{subset_str}seed{self.seed}.json"

        # Prepare simplified JSON (remove detailed per-question data for size)
        simplified_results = {}
        for method, result in self.results.items():
            simplified_results[method] = {
                'method': result['method'],
                'overall_accuracy': result['overall_accuracy'],
                'total_correct': result['total_correct'],
                'total_questions': result['total_questions'],
                'seed': result['seed'],
                'category_accuracies': {}
            }

            for category, cat_data in result['category_results'].items():
                simplified_results[method]['category_accuracies'][category] = {
                    'accuracy': cat_data['correct'] / cat_data['total'] if cat_data['total'] > 0 else 0,
                    'correct': cat_data['correct'],
                    'total': cat_data['total']
                }

        with open(json_path, 'w') as f:
            json.dump(simplified_results, f, indent=2)
        print(f"\n✓ Saved results: {json_path}")

        # Generate markdown summary
        md_lines = [
            f"# MMLU {'Subset ' if self.subset else ''}Evaluation Results (Seed {self.seed})",
            "",
            "## Overall Summary",
            "",
            "| Method | Overall Accuracy | Δ vs Base |",
            "|--------|------------------|-----------|"
        ]

        base_acc = self.results['base']['overall_accuracy']
        for method_name in ['base', 'prompt_persona', 'proposed']:
            result = self.results[method_name]
            acc = result['overall_accuracy']
            delta = acc - base_acc
            delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"

            md_lines.append(f"| {method_name.replace('_', ' ').title()} | {acc:.3f} | {delta_str} |")

        md_lines.extend(["", "## Category Breakdown", ""])

        # Category table
        md_lines.append("| Method | STEM | Humanities |")
        md_lines.append("|--------|------|------------|")

        for method_name in ['base', 'prompt_persona', 'proposed']:
            result = self.results[method_name]
            cat_accs = simplified_results[method_name]['category_accuracies']

            stem_acc = cat_accs.get('STEM', {}).get('accuracy', 0)
            hum_acc = cat_accs.get('Humanities', {}).get('accuracy', 0)

            md_lines.append(f"| {method_name.replace('_', ' ').title()} | {stem_acc:.3f} | {hum_acc:.3f} |")

        md_lines.extend(["", "## Interpretation", ""])

        # Add interpretation
        proposed_delta = self.results['proposed']['overall_accuracy'] - base_acc
        if abs(proposed_delta) < 0.03:
            md_lines.append("✓ **Proposed method maintains general capability** (±3% of baseline)")
        elif proposed_delta < -0.06:
            md_lines.append("⚠ **Proposed method significantly reduces capability** - may need adjustment")
        elif proposed_delta < -0.03:
            md_lines.append("⚠ **Proposed method shows moderate capability degradation** - acceptable trade-off for persona-fit")
        else:
            md_lines.append("✓ **Proposed method improves or maintains capability**")

        md_path = output_dir / f"mmlu_{subset_str}seed{self.seed}.md"
        with open(md_path, 'w') as f:
            f.write('\n'.join(md_lines))
        print(f"✓ Saved summary: {md_path}")


def run_single_persona(persona_id: str, seed: int, subset: bool, num_samples_per_subject: int):
    """Run MMLU evaluation for a single persona."""
    print(f"\n{'='*80}")
    print(f"Phase 2: MMLU Evaluation - {persona_id}")
    print(f"Seed: {seed}")
    print(f"Subset: {subset}")
    print(f"{'='*80}\n")

    evaluator = MMLUEvaluator(
        persona_id=persona_id,
        seed=seed,
        subset=subset,
        num_samples_per_subject=num_samples_per_subject
    )

    results = evaluator.run_all_methods()
    evaluator.save_results()

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--persona-id', type=str, help='Single persona ID')
    parser.add_argument('--persona-list', type=str, help='Path to personas.yaml')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--subset', action='store_true', help='Use subset (4 STEM + 4 Humanities)')
    parser.add_argument('--num-samples-per-subject', type=int, default=None,
                       help='Number of samples per subject (default: all)')

    args = parser.parse_args()

    # Get persona list
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from persona_opt.utils.persona_list_loader import get_persona_list_from_args
    persona_list = get_persona_list_from_args(args)

    print(f"\n{'='*80}")
    print(f"MMLU Multi-Persona Pipeline")
    print(f"Personas: {', '.join(persona_list)}")
    print(f"{'='*80}\n")

    # Run for each persona
    for persona_id in persona_list:
        print(f"\n{'#'*80}")
        print(f"# Processing: {persona_id}")
        print(f"{'#'*80}\n")

        run_single_persona(
            persona_id=persona_id,
            seed=args.seed,
            subset=args.subset,
            num_samples_per_subject=args.num_samples_per_subject
        )

    print(f"\n{'='*80}")
    print("All Personas Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
