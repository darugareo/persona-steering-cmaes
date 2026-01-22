"""
Internal steering module for Llama-3-8B using HF Transformers.
Allows direct manipulation of hidden states (residual stream) at specified layers.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List
import warnings


class Llama3ActivationSteerer:
    """
    Steers Llama-3-8B's internal activations by adding steering vectors
    to the residual stream at a specified layer.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        target_layer: int = 22,
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            target_layer: Which layer to apply steering (0-31 for Llama-3-8B)
            device: Device to load model on
            torch_dtype: Data type for model weights
        """
        self.model_name = model_name
        self.target_layer = target_layer
        self.device = device
        self.torch_dtype = torch_dtype

        # Load model and tokenizer
        print(f"Loading {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.model.eval()

        # Clear generation config to avoid conflicts
        self.model.generation_config.do_sample = None
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None

        # Steering state
        self.steering_vector: Optional[torch.Tensor] = None
        self.alpha: float = 1.0
        self.hook_handle = None

        print(f"Model loaded. Total layers: {len(self.model.model.layers)}")
        print(f"Hidden size: {self.model.config.hidden_size}")

    def _steering_hook(self, module, input_tuple, output):
        """
        Hook function that adds steering vector to residual stream.

        For Llama architecture:
        - Each layer output is a tuple (hidden_states, ...)
        - We modify hidden_states by adding the steering vector
        """
        if self.steering_vector is None:
            return output

        # Output is tuple: (hidden_states, *optional_outputs)
        hidden_states = output[0]  # Shape: (batch_size, seq_len, hidden_size)

        # Add steering vector (broadcast across batch and sequence)
        # steering_vector shape: (hidden_size,)
        steering_vec = self.steering_vector.to(
            hidden_states.device, dtype=hidden_states.dtype
        )
        # Reshape to (1, 1, hidden_size) for proper broadcasting
        steering_vec = steering_vec.view(1, 1, -1)
        steered_hidden_states = hidden_states + self.alpha * steering_vec

        # Return modified output
        if isinstance(output, tuple):
            return (steered_hidden_states,) + output[1:]
        else:
            return steered_hidden_states

    def register_hooks(
        self,
        steering_vector: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        multi_trait_vectors: Optional[dict] = None,
        trait_weights: Optional[dict] = None
    ):
        """
        Register steering hook at target layer.

        Supports two modes:
        1. Single-trait steering: Provide steering_vector + alpha
        2. Multi-trait steering: Provide multi_trait_vectors + trait_weights

        Args:
            steering_vector: Single vector to add (shape: [hidden_size])
                           If None, disables steering (unless multi_trait mode)
            alpha: Scaling factor for single steering vector
            multi_trait_vectors: Dict of {trait_name: steering_vector_tensor}
                                For multi-trait steering
            trait_weights: Dict of {trait_name: weight_value}
                          Weights for combining multiple trait vectors
        """
        # Remove existing hook if any
        self.remove_hooks()

        # Multi-trait mode
        if multi_trait_vectors is not None and trait_weights is not None:
            # Combine multiple trait vectors with weights
            expected_dim = self.model.config.hidden_size
            combined_vector = torch.zeros(expected_dim, dtype=self.torch_dtype, device=self.device)

            total_weight = 0.0
            for trait_name, trait_vector in multi_trait_vectors.items():
                weight = trait_weights.get(trait_name, 0.0)
                if weight == 0.0:
                    continue

                # Validate shape
                if trait_vector.shape != (expected_dim,):
                    raise ValueError(
                        f"Trait vector '{trait_name}' shape {trait_vector.shape} "
                        f"doesn't match expected ({expected_dim},)"
                    )

                combined_vector = combined_vector + weight * trait_vector.to(
                    device=self.device, dtype=self.torch_dtype
                )
                total_weight += abs(weight)

            if total_weight == 0.0:
                print("Warning: All trait weights are zero, disabling steering")
                self.steering_vector = None
                return

            self.steering_vector = combined_vector
            self.alpha = 1.0  # Weights already applied

            # Register forward hook
            target_module = self.model.model.layers[self.target_layer]
            self.hook_handle = target_module.register_forward_hook(self._steering_hook)

            print(f"Registered multi-trait steering hook at layer {self.target_layer}")
            print(f"  Active traits: {[k for k, v in trait_weights.items() if v != 0.0]}")
            print(f"  Weights: {trait_weights}")
            print(f"  Combined vector norm: {combined_vector.norm().item():.4f}")
            return

        # Single-trait mode
        if steering_vector is None:
            self.steering_vector = None
            return

        # Validate steering vector shape
        expected_dim = self.model.config.hidden_size
        if steering_vector.shape != (expected_dim,):
            raise ValueError(
                f"Steering vector shape {steering_vector.shape} doesn't match "
                f"expected ({expected_dim},)"
            )

        self.steering_vector = steering_vector
        self.alpha = alpha

        # Register forward hook on target layer
        target_module = self.model.model.layers[self.target_layer]
        self.hook_handle = target_module.register_forward_hook(self._steering_hook)

        print(f"Registered steering hook at layer {self.target_layer} with alpha={alpha}")

    def remove_hooks(self):
        """Remove all registered hooks."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        self.steering_vector = None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        do_sample: bool = False,
        top_p: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate text with current steering configuration.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            do_sample: Whether to use sampling
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            Generated text (prompt + completion)
        """
        # Format prompt with chat template if available
        if self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            # For greedy decoding (temperature=0), don't use sampling
            if temperature == 0.0 and not do_sample:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    **kwargs
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=True,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    **kwargs
                )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text

    def batch_generate(
        self,
        prompts: list,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        do_sample: bool = False,
        top_p: float = 1.0,
        steering_vector=None,
        layer=None,
        alpha=None,
        **kwargs
    ) -> list:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Nucleus sampling parameter
            steering_vector: Optional steering vector to apply
            layer: Layer to apply steering (if different from default)
            alpha: Steering strength (if different from current)
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        # Apply steering if provided
        original_layer = None
        if steering_vector is not None:
            if layer is not None:
                original_layer = self.target_layer
                self.target_layer = layer
            self.register_hooks(steering_vector=steering_vector, alpha=alpha if alpha is not None else self.alpha)

        try:
            results = []
            for prompt in prompts:
                result = self.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    **kwargs
                )
                results.append(result)
            return results
        finally:
            # Restore original layer if changed
            if steering_vector is not None and layer is not None:
                self.target_layer = original_layer
                self.remove_hooks()

    def get_hidden_states(
        self,
        prompt: str,
        layer: Optional[int] = None
    ) -> torch.Tensor:
        """
        Extract hidden states at specified layer for given prompt.

        Args:
            prompt: Input text
            layer: Which layer to extract from (default: target_layer)

        Returns:
            Hidden states tensor of shape (seq_len, hidden_size)
        """
        if layer is None:
            layer = self.target_layer

        # Format prompt
        if self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt"
        ).to(self.device)

        # Forward pass with output_hidden_states
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # Extract hidden states at specified layer
        # outputs.hidden_states is tuple of (num_layers + 1) tensors
        # Index 0 is embedding, index i+1 is layer i output
        hidden_states = outputs.hidden_states[layer + 1]  # Shape: (1, seq_len, hidden_size)

        return hidden_states.squeeze(0)  # Return (seq_len, hidden_size)

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


def build_contrast_steering_vector(
    steerer: Llama3ActivationSteerer,
    positive_prompts: List[str],
    negative_prompts: List[str],
    layer: Optional[int] = None,
    aggregate: str = "mean"
) -> torch.Tensor:
    """
    Build a steering vector from contrast between positive and negative examples.

    This follows the SVT (Small Vectors Training) approach:
    1. Get hidden states for positive examples (e.g., "other-focused")
    2. Get hidden states for negative examples (e.g., "self-focused")
    3. Compute difference: steering_vec = mean(positive) - mean(negative)

    Args:
        steerer: Llama3ActivationSteerer instance
        positive_prompts: List of prompts representing positive direction
        negative_prompts: List of prompts representing negative direction
        layer: Which layer to extract from (default: steerer.target_layer)
        aggregate: How to aggregate across tokens ("mean" or "last")

    Returns:
        Steering vector of shape (hidden_size,)
    """
    if layer is None:
        layer = steerer.target_layer

    print(f"Building steering vector from {len(positive_prompts)} positive "
          f"and {len(negative_prompts)} negative examples at layer {layer}...")

    # Collect hidden states for positive examples
    positive_hiddens = []
    for prompt in positive_prompts:
        hidden = steerer.get_hidden_states(prompt, layer=layer)
        # Aggregate across sequence dimension
        if aggregate == "mean":
            hidden_agg = hidden.mean(dim=0)  # (hidden_size,)
        elif aggregate == "last":
            hidden_agg = hidden[-1]  # (hidden_size,)
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")
        positive_hiddens.append(hidden_agg)

    # Collect hidden states for negative examples
    negative_hiddens = []
    for prompt in negative_prompts:
        hidden = steerer.get_hidden_states(prompt, layer=layer)
        if aggregate == "mean":
            hidden_agg = hidden.mean(dim=0)
        elif aggregate == "last":
            hidden_agg = hidden[-1]
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")
        negative_hiddens.append(hidden_agg)

    # Stack and compute mean across examples
    positive_mean = torch.stack(positive_hiddens).mean(dim=0)  # (hidden_size,)
    negative_mean = torch.stack(negative_hiddens).mean(dim=0)  # (hidden_size,)

    # Compute steering vector as difference
    steering_vector = positive_mean - negative_mean

    print(f"Steering vector built. Norm: {steering_vector.norm().item():.4f}")

    return steering_vector
