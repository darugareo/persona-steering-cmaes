"""
Multi-Turn Persona Stability Evaluation.

Tests whether persona is maintained across multi-turn conversations.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch

from .utils import (
    load_optimization_results,
    build_combined_steering_vector,
    save_evaluation_results,
    EvaluationConfig
)


class MultiTurnEvaluator:
    """Evaluates persona stability in multi-turn conversations."""

    def __init__(
        self,
        config: EvaluationConfig,
        steerer,
        evaluator,
        optimization_dir: str = "persona-opt",
        vectors_dir: str = "data/steering_vectors_v2"
    ):
        """
        Initialize evaluator.

        Args:
            config: Evaluation configuration
            steerer: Llama3ActivationSteerer instance
            evaluator: PersonaAwareEvaluator instance
            optimization_dir: Directory with optimization results
            vectors_dir: Directory with steering vectors
        """
        self.config = config
        self.steerer = steerer
        self.evaluator = evaluator
        self.optimization_dir = optimization_dir
        self.vectors_dir = vectors_dir

        # Load optimization results
        self.opt_results = load_optimization_results(config.persona_id, optimization_dir)
        self.weights = self.opt_results['weights']
        self.alpha = self.opt_results.get('alpha', 2.0)
        self.optimized_layer = self.opt_results.get('layer', config.layer)

    def evaluate(
        self,
        initial_prompts: List[str],
        num_turns: int = 5,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Run multi-turn evaluation.

        Args:
            initial_prompts: Starting prompts for conversations
            num_turns: Number of conversation turns
            output_dir: Output directory for results

        Returns:
            Results dictionary
        """
        print(f"[MultiTurn] Evaluating {self.config.persona_id}")
        print(f"[MultiTurn] Conversations: {len(initial_prompts)}")
        print(f"[MultiTurn] Turns per conversation: {num_turns}")

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

        # Run multi-turn conversations
        all_conversations = []
        turn_scores = {i: [] for i in range(num_turns)}

        for conv_idx, initial_prompt in enumerate(initial_prompts):
            print(f"\n[MultiTurn] Conversation {conv_idx + 1}/{len(initial_prompts)}")

            conversation = self._run_conversation(
                initial_prompt,
                num_turns,
                steering_vector
            )

            all_conversations.append(conversation)

            # Collect scores per turn
            for turn_idx, turn_data in enumerate(conversation['turns']):
                turn_scores[turn_idx].append(turn_data['persona_fit'])

        # Compute statistics
        turn_statistics = {}
        for turn_idx in range(num_turns):
            scores = turn_scores[turn_idx]
            turn_statistics[turn_idx] = {
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores))
            }

        # Detect persona drift
        first_turn_mean = turn_statistics[0]['mean_score']
        last_turn_mean = turn_statistics[num_turns - 1]['mean_score']
        drift = first_turn_mean - last_turn_mean
        drift_rate = drift / num_turns

        results = {
            'evaluation_type': 'multi_turn_stability',
            'persona_id': self.config.persona_id,
            'layer': self.optimized_layer,
            'alpha': self.alpha,
            'timestamp': datetime.now().isoformat(),
            'weights': self.weights,
            'config': {
                'num_conversations': len(initial_prompts),
                'num_turns': num_turns
            },
            'conversations': all_conversations,
            'turn_statistics': turn_statistics,
            'drift_analysis': {
                'first_turn_mean': float(first_turn_mean),
                'last_turn_mean': float(last_turn_mean),
                'total_drift': float(drift),
                'drift_per_turn': float(drift_rate),
                'stable': abs(drift) < 0.5
            },
            'summary': {
                'num_turns': num_turns,
                'first_turn_score': f"{first_turn_mean:.2f}",
                'last_turn_score': f"{last_turn_mean:.2f}",
                'drift': f"{drift:+.2f}",
                'stable': "Yes" if abs(drift) < 0.5 else "No"
            }
        }

        # Generate plots
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            fig_path = self._plot_results(results, output_path)
            save_evaluation_results(results, output_dir, {'Multi-Turn Stability': str(fig_path)})

        return results

    def _run_conversation(
        self,
        initial_prompt: str,
        num_turns: int,
        steering_vector: torch.Tensor
    ) -> Dict:
        """Run a single multi-turn conversation."""
        conversation = {
            'initial_prompt': initial_prompt,
            'turns': []
        }

        current_prompt = initial_prompt
        conversation_history = []

        for turn_idx in range(num_turns):
            # Generate baseline response (without steering)
            self.steerer.remove_hooks()
            baseline_response = self.steerer.generate(current_prompt)

            # Generate steered response
            self.steerer.register_hooks(steering_vector=steering_vector, alpha=self.alpha)
            steered_response = self.steerer.generate(current_prompt)

            # Evaluate this turn
            eval_result = self.evaluator.evaluate_with_persona_judge(
                prompt=current_prompt,
                baseline_response=baseline_response,
                steered_response=steered_response
            )

            turn_data = {
                'turn': turn_idx,
                'prompt': current_prompt,
                'baseline': baseline_response,
                'steered': steered_response,
                'persona_fit': eval_result['persona_fit'],
                'winner': eval_result['winner']
            }

            conversation['turns'].append(turn_data)

            # Update conversation history with steered response
            conversation_history.append({
                'role': 'user',
                'content': current_prompt
            })
            conversation_history.append({
                'role': 'assistant',
                'content': steered_response
            })

            # Generate follow-up prompt for next turn
            if turn_idx < num_turns - 1:
                current_prompt = self._generate_followup_prompt(
                    current_prompt,
                    steered_response
                )

        return conversation

    def _generate_followup_prompt(self, previous_prompt: str, response: str) -> str:
        """Generate a natural follow-up prompt based on the conversation."""
        # Simple follow-up prompts (can be made more sophisticated)
        followups = [
            "Can you tell me more about that?",
            "What do you think about it?",
            "How did that make you feel?",
            "What happened next?",
            "Why do you think that is?"
        ]

        # Use hash to deterministically select follow-up
        import hashlib
        hash_val = int(hashlib.md5((previous_prompt + response).encode()).hexdigest(), 16)
        return followups[hash_val % len(followups)]

    def _plot_results(self, results: Dict, output_dir: Path) -> Path:
        """Generate plot of persona stability over turns."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        turns = list(range(results['config']['num_turns']))
        turn_stats = results['turn_statistics']

        # Plot 1: Mean score per turn with error bars
        means = [turn_stats[t]['mean_score'] for t in turns]
        stds = [turn_stats[t]['std_score'] for t in turns]

        ax1.plot(turns, means, marker='o', linewidth=2, markersize=8, color='#3498db')
        ax1.fill_between(turns,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.2, color='#3498db')
        ax1.axhline(y=5.0, color='green', linestyle=':', alpha=0.5, label='Perfect Score')
        ax1.set_xlabel('Turn Number', fontsize=12)
        ax1.set_ylabel('Persona Fit Score', fontsize=12)
        ax1.set_title('Multi-Turn Persona Stability', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 5.5)
        ax1.grid(alpha=0.3)
        ax1.legend()

        # Plot 2: Score distribution per turn (box plot)
        score_data = []
        for conv in results['conversations']:
            for turn in conv['turns']:
                score_data.append((turn['turn'], turn['persona_fit']))

        # Organize by turn
        turn_scores = {t: [] for t in turns}
        for turn_idx, score in score_data:
            turn_scores[turn_idx].append(score)

        ax2.boxplot([turn_scores[t] for t in turns], positions=turns, widths=0.6)
        ax2.axhline(y=5.0, color='green', linestyle=':', alpha=0.5, label='Perfect Score')
        ax2.set_xlabel('Turn Number', fontsize=12)
        ax2.set_ylabel('Persona Fit Score', fontsize=12)
        ax2.set_title('Score Distribution per Turn', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 5.5)
        ax2.grid(alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        fig_path = output_dir / 'plot.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        return fig_path
