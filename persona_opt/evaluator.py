"""
PersonaAwareEvaluator - Wrapper for batch evaluation with persona-aware judge.

Provides a unified interface for evaluation modules.
"""

from typing import Dict, List, Optional
from persona_opt.persona_judge_evaluator import evaluate_with_persona_judge
from persona_opt.evaluation.utils import load_persona_profile


class PersonaAwareEvaluator:
    """Wrapper for persona-aware judge evaluations."""

    def __init__(
        self,
        persona_profile: Optional[Dict] = None,
        persona_id: Optional[str] = None,
        judge_model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        base_dir: str = "personas",
        # Metadata for logging
        method_name: Optional[str] = None,
        seed: Optional[int] = None,
        layer: Optional[int] = None,
        alpha: Optional[float] = None,
        weights: Optional[list] = None,
        experiment: Optional[str] = None  # New: Phase 2 experiment name
    ):
        """
        Initialize evaluator.

        Args:
            persona_profile: Persona profile dict (loaded externally)
            persona_id: Persona ID (will load profile if persona_profile not provided)
            judge_model: Judge LLM model name
            temperature: Sampling temperature
            base_dir: Base directory for persona data
            method_name: Method name for logging (e.g., "proposed", "meandiff")
            seed: Random seed for logging
            layer: Layer number for logging
            alpha: Alpha parameter for logging
            weights: Weight vector for logging
        """
        if persona_profile:
            self.persona_profile = persona_profile
            self.persona_id = persona_profile.get('persona_id')
        elif persona_id:
            self.persona_id = persona_id
            self.persona_profile = load_persona_profile(persona_id)
        else:
            raise ValueError("Must provide either persona_profile or persona_id")

        self.judge_model = judge_model
        self.temperature = temperature
        self.base_dir = base_dir

        # Metadata for logging
        self.method_name = method_name
        self.seed = seed
        self.layer = layer
        self.alpha = alpha
        self.weights = weights
        self.experiment = experiment

    def evaluate_with_persona_judge(
        self,
        prompt: str,
        baseline_response: str,
        steered_response: str
    ) -> Dict:
        """
        Evaluate single prompt with persona judge.

        Args:
            prompt: User prompt
            baseline_response: Baseline model response
            steered_response: Steered model response

        Returns:
            {
                "winner": "baseline" or "steered" or "tie",
                "persona_fit": float (1-5 score for steered),
                "baseline_fit": float (1-5 score for baseline),
                "confidence": int (1-5),
                "explanation": str
            }
        """
        result = evaluate_with_persona_judge(
            persona_id=self.persona_id,
            prompt=prompt,
            response_a=baseline_response,
            response_b=steered_response,
            trait_name="Overall Persona Fit",
            trait_direction="matches persona style and values",
            base_dir=self.base_dir,
            model=self.judge_model,
            temperature=self.temperature,
            # Pass metadata for logging
            method_name=self.method_name,
            seed=self.seed,
            layer=self.layer,
            alpha=self.alpha,
            weights=self.weights,
            experiment=self.experiment
        )

        # Convert to standard format
        winner_map = {"A": "baseline", "B": "steered", "tie": "tie"}

        return {
            "winner": winner_map.get(result.get("winner", "tie"), "tie"),
            "persona_fit": float(result.get("persona_fit_score_b", 3.0)),
            "baseline_fit": float(result.get("persona_fit_score_a", 3.0)),
            "confidence": int(result.get("confidence", 3)),
            "explanation": result.get("explanation", "")
        }

    def batch_evaluate(
        self,
        prompts: List[str],
        baseline_responses: List[str],
        steered_responses: List[str]
    ) -> List[Dict]:
        """
        Evaluate multiple prompts in batch.

        Args:
            prompts: List of prompts
            baseline_responses: List of baseline responses
            steered_responses: List of steered responses

        Returns:
            List of evaluation results
        """
        results = []

        for i, (prompt, baseline, steered) in enumerate(zip(prompts, baseline_responses, steered_responses)):
            print(f"  Evaluating {i+1}/{len(prompts)}...")
            result = self.evaluate_with_persona_judge(prompt, baseline, steered)
            results.append(result)

        return results
