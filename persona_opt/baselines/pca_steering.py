"""
Baseline 4: PCA-based Steering
Extracts principal components from persona activations.
Uses top PC as steering direction.
"""

from typing import List, Dict, Any, Optional
import torch
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from persona_opt.evaluation.utils import load_persona_profile


class PCASteeringMethod:
    """
    PCA-based steering: extracts principal components from persona activations.
    """

    def __init__(
        self,
        steerer,
        persona_id: str,
        layer: int = 20,
        alpha: float = 2.0,
        n_components: int = 5,
        persona_profile: Optional[Dict] = None,
        extraction_prompts: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Args:
            steerer: Llama3ActivationSteerer instance
            persona_id: Persona identifier
            layer: Target layer for steering
            alpha: Steering strength
            n_components: Number of PCA components
            persona_profile: Preloaded persona profile
            extraction_prompts: Prompts for PCA extraction
            **kwargs: Additional configuration
        """
        self.steerer = steerer
        self.persona_id = persona_id
        self.layer = layer
        self.alpha = alpha
        self.n_components = n_components
        self.method_name = "pca"

        # Load persona profile
        if persona_profile is None:
            persona_profile = load_persona_profile(persona_id)
        self.persona_profile = persona_profile

        # Use default prompts if not provided
        if extraction_prompts is None:
            extraction_prompts = self._get_default_prompts()
        self.extraction_prompts = extraction_prompts

        # Compute PCA components
        self.pca, self.components = self._compute_pca_components()

        # Use first component as steering vector
        self.steering_vector = self.components[0]

    def _get_default_prompts(self) -> List[str]:
        """Get default prompts for PCA extraction."""
        return [
            "Tell me about yourself.",
            "How do you communicate with others?",
            "What's your personality like?",
            "How do you approach problems?",
            "What are your values?",
            "How do you express yourself?",
            "What's your communication style?",
            "How do you interact with people?",
            "What makes you unique?",
            "How would you describe yourself?",
        ]

    def _compute_pca_components(self) -> tuple:
        """
        Compute PCA components from persona activations.

        Returns:
            (pca_model, components_tensor)
        """
        print(f"[PCA] Extracting {self.n_components} components at layer {self.layer}...")

        # Collect activations
        activations = []
        for prompt in self.extraction_prompts:
            hidden = self.steerer.get_hidden_states(prompt, layer=self.layer)
            # Take mean across sequence, convert to float32 for numpy
            activations.append(hidden.mean(dim=0).float().cpu().numpy())

        # Stack activations: (n_prompts, hidden_size)
        activations_matrix = np.stack(activations)

        # Fit PCA
        pca = PCA(n_components=self.n_components)
        pca.fit(activations_matrix)

        # Extract components as tensors
        components = [
            torch.tensor(comp, dtype=torch.float32)
            for comp in pca.components_
        ]

        # Print variance explained
        var_explained = pca.explained_variance_ratio_
        print(f"[PCA] Variance explained by top {self.n_components} components:")
        for i, var in enumerate(var_explained):
            print(f"  PC{i+1}: {var*100:.2f}%")

        print(f"[PCA] Using PC1 as steering vector (norm: {components[0].norm().item():.4f})")

        return pca, components

    def generate(self, prompt: str, **generation_kwargs) -> str:
        """
        Generate response with PCA steering.

        Args:
            prompt: Input prompt
            **generation_kwargs: Generation parameters

        Returns:
            Generated response
        """
        # Apply steering with first PC
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
        var_explained = self.pca.explained_variance_ratio_[0] if hasattr(self.pca, 'explained_variance_ratio_') else 0.0

        return {
            'method': self.method_name,
            'persona_id': self.persona_id,
            'layer': self.layer,
            'alpha': self.alpha,
            'n_components': self.n_components,
            'description': 'PCA-based activation steering',
            'vector_norm': self.steering_vector.norm().item(),
            'variance_explained_pc1': float(var_explained),
            'steering': 'activation_based',
        }

    def __repr__(self):
        return f"PCASteeringMethod(persona_id={self.persona_id}, layer={self.layer}, n_components={self.n_components})"
