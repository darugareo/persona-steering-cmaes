"""
Proposed Method: SVD + CMA-ES optimized steering
Uses pre-optimized weights from persona-opt/{persona_id}/best_weights.json
"""

import json
import torch
from pathlib import Path
from typing import Dict, List
import numpy as np


class ProposedMethod:
    """Proposed method using SVD vectors with CMA-ES optimized weights."""

    def __init__(
        self,
        steerer,
        persona_id: str,
        layer: int = 20,
        alpha: float = 2.0,
    ):
        self.steerer = steerer
        self.persona_id = persona_id
        self.layer = layer
        self.alpha = alpha

        # Load optimized weights - NO FALLBACK TO EQUAL WEIGHTS
        weights_file = Path(f"persona-opt/{persona_id}/best_weights.json")
        if not weights_file.exists():
            raise FileNotFoundError(
                f"Optimized weights file not found: {weights_file}\n"
                f"CMA-ES optimization must be run first for persona {persona_id}.\n"
                f"Expected file location: persona-opt/{persona_id}/best_weights.json"
            )

        with open(weights_file, 'r') as f:
            weights_data = json.load(f)
            if isinstance(weights_data, dict):
                # Handle dict format with R1-R5 keys
                if 'R1' in weights_data:
                    self.weights = [
                        weights_data['R1'],
                        weights_data['R2'],
                        weights_data['R3'],
                        weights_data['R4'],
                        weights_data['R5']
                    ]
                elif 'weights' in weights_data:
                    self.weights = weights_data['weights']
                else:
                    raise ValueError(f"Invalid weights format in {weights_file}")
            elif isinstance(weights_data, list):
                self.weights = weights_data
            else:
                raise ValueError(f"Invalid weights format in {weights_file}")

        # Load SVD vectors from data/steering_vectors_v2
        self.trait_vectors = {}
        for trait in ["R1", "R2", "R3", "R4", "R5"]:
            vector_file = Path(f"data/steering_vectors_v2/{trait}/layer{layer}_svd.pt")
            if not vector_file.exists():
                raise FileNotFoundError(f"SVD vector not found: {vector_file}")

            vector_data = torch.load(vector_file, map_location='cpu')
            if isinstance(vector_data, torch.Tensor):
                self.trait_vectors[trait] = vector_data
            elif isinstance(vector_data, dict):
                self.trait_vectors[trait] = vector_data['vector']

        # Build combined steering vector with optimized weights
        self.steering_vector = self._build_combined_vector()

        print(f"✓ Optimized CMA-ES weights successfully loaded.")
        print(f"[Proposed] Weights: {self.weights}")
        print(f"[Proposed] Using layer {layer} with alpha {alpha}")

    def _build_combined_vector(self):
        """Build combined vector from weighted trait vectors."""
        combined = torch.zeros_like(list(self.trait_vectors.values())[0])
        for i, trait in enumerate(["R1", "R2", "R3", "R4", "R5"]):
            combined += self.weights[i] * self.trait_vectors[trait]
        return combined

    def get_config(self) -> Dict:
        """Return method configuration."""
        return {
            'method': 'proposed',
            'description': 'SVD + CMA-ES + LLM Judge',
            'layer': self.layer,
            'alpha': self.alpha,
            'weights': self.weights,
            'steering': 'activation_based'
        }

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response with optimized steering."""
        # Apply combined steering vector
        self.steerer.register_hooks(
            steering_vector=self.steering_vector,
            alpha=self.alpha
        )

        response = self.steerer.generate(prompt, **kwargs)

        return response

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts."""
        responses = []

        for prompt in prompts:
            response = self.generate(prompt, **kwargs)
            responses.append(response)

        return responses
