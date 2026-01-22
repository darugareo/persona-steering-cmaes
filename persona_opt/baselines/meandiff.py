"""
Baseline 3: Mean Difference Steering
Computes mean difference between persona and non-persona activations.
Classic contrastive activation steering approach.
"""

from typing import List, Dict, Any, Optional
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from persona_opt.evaluation.utils import load_persona_profile


class MeanDiffMethod:
    """
    Mean difference steering: direction = mean(persona) - mean(non-persona)
    """

    def __init__(
        self,
        steerer,
        persona_id: str,
        layer: int = 20,
        alpha: float = 2.0,
        persona_profile: Optional[Dict] = None,
        contrast_prompts: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Args:
            steerer: Llama3ActivationSteerer instance
            persona_id: Persona identifier
            layer: Target layer for steering
            alpha: Steering strength
            persona_profile: Preloaded persona profile
            contrast_prompts: Prompts for computing mean difference
            **kwargs: Additional configuration
        """
        self.steerer = steerer
        self.persona_id = persona_id
        self.layer = layer
        self.alpha = alpha
        self.method_name = "meandiff"

        # Load persona profile
        if persona_profile is None:
            persona_profile = load_persona_profile(persona_id)
        self.persona_profile = persona_profile

        # Use default contrast prompts if not provided
        if contrast_prompts is None:
            contrast_prompts = self._get_default_contrast_prompts()
        self.contrast_prompts = contrast_prompts

        # Compute steering vector
        self.steering_vector = self._compute_meandiff_vector()

    def _get_default_contrast_prompts(self) -> Dict[str, List[str]]:
        """Get default prompts for persona vs non-persona contrast."""
        # Generic prompts that should exhibit persona traits
        persona_prompts = [
            "Tell me about yourself.",
            "How do you approach conversations?",
            "What's your communication style?",
            "How do you help people?",
            "What's important to you in interactions?",
        ]

        # Generic neutral prompts
        neutral_prompts = [
            "What is 2+2?",
            "Define the word 'algorithm'.",
            "List three primary colors.",
            "What is the capital of France?",
            "Explain photosynthesis.",
        ]

        return {
            'persona': persona_prompts,
            'neutral': neutral_prompts
        }

    def _compute_meandiff_vector(self) -> torch.Tensor:
        """
        Compute mean difference steering vector.

        Returns:
            Steering vector of shape (hidden_size,)
        """
        print(f"[MeanDiff] Computing steering vector at layer {self.layer}...")

        # Get activations for persona prompts
        persona_activations = []
        for prompt in self.contrast_prompts['persona']:
            hidden = self.steerer.get_hidden_states(prompt, layer=self.layer)
            # Take mean across sequence
            persona_activations.append(hidden.mean(dim=0))

        # Get activations for neutral prompts
        neutral_activations = []
        for prompt in self.contrast_prompts['neutral']:
            hidden = self.steerer.get_hidden_states(prompt, layer=self.layer)
            neutral_activations.append(hidden.mean(dim=0))

        # Compute means
        persona_mean = torch.stack(persona_activations).mean(dim=0)
        neutral_mean = torch.stack(neutral_activations).mean(dim=0)

        # Steering vector is the difference
        steering_vector = persona_mean - neutral_mean

        print(f"[MeanDiff] Vector norm: {steering_vector.norm().item():.4f}")

        return steering_vector

    def generate(self, prompt: str, **generation_kwargs) -> str:
        """
        Generate response with mean difference steering.

        Args:
            prompt: Input prompt
            **generation_kwargs: Generation parameters

        Returns:
            Generated response
        """
        # Apply steering
        self.steerer.register_hooks(
            steering_vector=self.steering_vector,
            alpha=self.alpha
        )

        # Generate
        response = self.steerer.generate(prompt, **generation_kwargs)

        return response

    def batch_generate(self, prompts: List[str], **generation_kwargs) -> List[str]:
        """
        Generate multiple responses with steering.

        Args:
            prompts: List of input prompts
            **generation_kwargs: Generation parameters

        Returns:
            List of generated responses
        """
        return self.steerer.batch_generate(
            prompts,
            steering_vector=self.steering_vector,
            layer=self.layer,
            alpha=self.alpha,
            **generation_kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Get method configuration."""
        return {
            'method': self.method_name,
            'persona_id': self.persona_id,
            'layer': self.layer,
            'alpha': self.alpha,
            'description': 'Mean difference activation steering',
            'vector_norm': self.steering_vector.norm().item(),
            'steering': 'activation_based',
        }

    def __repr__(self):
        return f"MeanDiffMethod(persona_id={self.persona_id}, layer={self.layer}, alpha={self.alpha})"
