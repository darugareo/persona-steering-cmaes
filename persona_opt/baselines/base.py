"""
Baseline 1: Base Method (No Steering)
Pure model responses without any persona steering.
"""

from typing import List, Dict, Any, Optional
import torch
from pathlib import Path


class BaseMethod:
    """
    Baseline method: No steering applied.
    Generates responses using the base model without any modifications.
    """

    def __init__(
        self,
        steerer,
        persona_id: str,
        **kwargs
    ):
        """
        Args:
            steerer: Llama3ActivationSteerer instance
            persona_id: Persona identifier
            **kwargs: Additional configuration
        """
        self.steerer = steerer
        self.persona_id = persona_id
        self.method_name = "base"

    def generate(self, prompt: str, **generation_kwargs) -> str:
        """
        Generate response without steering.

        Args:
            prompt: Input prompt
            **generation_kwargs: Generation parameters

        Returns:
            Generated response
        """
        # Remove any existing hooks
        self.steerer.remove_hooks()

        # Generate without steering
        response = self.steerer.generate(prompt, **generation_kwargs)

        return response

    def batch_generate(self, prompts: List[str], **generation_kwargs) -> List[str]:
        """
        Generate multiple responses without steering.

        Args:
            prompts: List of input prompts
            **generation_kwargs: Generation parameters

        Returns:
            List of generated responses
        """
        # Ensure no steering is applied
        self.steerer.remove_hooks()

        responses = []
        for prompt in prompts:
            response = self.steerer.generate(prompt, **generation_kwargs)
            responses.append(response)

        return responses

    def get_config(self) -> Dict[str, Any]:
        """Get method configuration."""
        return {
            'method': self.method_name,
            'persona_id': self.persona_id,
            'description': 'Baseline model without steering',
            'steering': None,
        }

    def __repr__(self):
        return f"BaseMethod(persona_id={self.persona_id})"
