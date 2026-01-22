"""
SVD-based Steering Vector Builder

Builds steering vectors using SVD/SVT-style contrastive activation extraction
for Llama-3-8B internal steering.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
from pathlib import Path
import json


class ActivationCollector:
    """Collects hidden state activations from specified transformer layers."""

    def __init__(self, model: nn.Module, tokenizer, layers: List[int], device: torch.device):
        """
        Initialize activation collector.

        Args:
            model: Transformer model
            tokenizer: Model tokenizer
            layers: List of layer indices to collect from
            device: Device to run model on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.device = device
        self.hooks = []
        self.storage = {layer: [] for layer in layers}

    def register_hooks(self):
        """Install forward hooks for each target layer."""
        self.storage = {layer: [] for layer in self.layers}

        for layer_idx in self.layers:
            module = self.model.model.layers[layer_idx]

            def hook_fn(module, input, output, layer=layer_idx):
                # Llama layer output is a tuple: (hidden_states,)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # hidden_states shape: (batch, seq, hidden_dim)
                # Mean pool over sequence dimension
                pooled = hidden_states.mean(dim=1)  # (batch, hidden_dim)

                # Store first batch element (we process one prompt at a time)
                self.storage[layer].append(pooled[0].detach().cpu())

            handle = module.register_forward_hook(hook_fn)
            self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def collect(self, prompts: List[str]):
        """
        Collect activations for a list of prompts.
        Processes one prompt at a time to ensure consistent number of activations.

        Args:
            prompts: List of text prompts
        """
        self.register_hooks()

        try:
            for prompt in prompts:
                # Tokenize single prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                # Forward pass
                with torch.no_grad():
                    self.model(**inputs)

        finally:
            self.remove_hooks()

    def get_activations(self, layer: int) -> List[torch.Tensor]:
        """
        Get collected activations for a specific layer.

        Args:
            layer: Layer index

        Returns:
            List of activation tensors, each of shape (hidden_dim,)
        """
        return self.storage[layer]


class SVDSteeringBuilder:
    """Builds steering vectors using SVD on contrastive activations."""

    def __init__(
        self,
        model_name: str,
        layers: List[int],
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None
    ):
        """
        Initialize SVD steering builder.

        Args:
            model_name: HuggingFace model name
            layers: List of layer indices to build vectors for
            torch_dtype: Model dtype (default: bfloat16)
            device: Device to use (default: auto-detect)
        """
        self.model_name = model_name
        self.layers = layers
        self.torch_dtype = torch_dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Dtype: {torch_dtype}")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=self.device if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

        # Storage for computed vectors
        self.vectors = {}
        self.positive_activations = {}
        self.negative_activations = {}

    def collect_activations(self, positive_prompts: List[str], negative_prompts: List[str]):
        """
        Collect activations for positive and negative prompt sets.

        Args:
            positive_prompts: List of persona-like or trait-positive prompts
            negative_prompts: List of persona-unlike or trait-negative prompts
        """
        print(f"\nCollecting activations for {len(positive_prompts)} positive prompts...")
        pos_collector = ActivationCollector(self.model, self.tokenizer, self.layers, self.device)
        pos_collector.collect(positive_prompts)

        print(f"Collecting activations for {len(negative_prompts)} negative prompts...")
        neg_collector = ActivationCollector(self.model, self.tokenizer, self.layers, self.device)
        neg_collector.collect(negative_prompts)

        # Store activations
        for layer in self.layers:
            self.positive_activations[layer] = pos_collector.get_activations(layer)
            self.negative_activations[layer] = neg_collector.get_activations(layer)

    def compute_svd_directions(self) -> Dict[int, torch.Tensor]:
        """
        Compute SVD-based steering directions for each layer.

        For each layer:
            1. Stack positive and negative activations
            2. Compute M = H_pos - H_neg
            3. Perform SVD: U, S, Vt = svd(M)
            4. Extract principal direction: d = Vt[0]
            5. Normalize: d = d / ||d||

        Returns:
            Dictionary mapping layer index to steering vector
        """
        print("\nComputing SVD directions...")

        for layer in self.layers:
            pos_acts = self.positive_activations[layer]
            neg_acts = self.negative_activations[layer]

            print(f"Layer {layer}: POS activations count = {len(pos_acts)}, NEG = {len(neg_acts)}")
            if len(pos_acts) > 0:
                print(f"Layer {layer}: First POS activation shape = {pos_acts[0].shape}")

            # Stack activations: (N, hidden_dim)
            H_pos = torch.stack(pos_acts)
            H_neg = torch.stack(neg_acts)

            print(f"Layer {layer}: H_pos shape = {H_pos.shape}, H_neg shape = {H_neg.shape}")

            # Compute contrast matrix
            M = H_pos - H_neg  # (N, hidden_dim)

            print(f"Layer {layer}: M shape = {M.shape}")

            # Convert to float32 for SVD (bfloat16 not supported on CPU)
            M = M.float()

            # Perform SVD
            U, S, Vt = torch.linalg.svd(M, full_matrices=False)

            # Extract principal component (first right singular vector)
            principal = Vt[0]  # (hidden_dim,)

            # Normalize
            principal = principal / principal.norm()

            self.vectors[layer] = principal

            print(f"Layer {layer}: principal direction norm = {principal.norm().item():.6f}")
            print(f"Layer {layer}: top 3 singular values = {S[:3].tolist()}")

        return self.vectors

    def save(self, save_dir: str):
        """
        Save steering vectors to disk.

        Args:
            save_dir: Directory to save vectors to
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving vectors to {save_path}")

        for layer, vector in self.vectors.items():
            output_file = save_path / f"layer{layer}_svd.pt"
            torch.save(vector, output_file)
            print(f"Saved layer {layer} → {output_file}")

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "layers": self.layers,
            "dtype": str(self.torch_dtype),
            "vector_dim": self.vectors[self.layers[0]].shape[0],
            "num_positive": len(self.positive_activations[self.layers[0]]),
            "num_negative": len(self.negative_activations[self.layers[0]])
        }

        metadata_file = save_path / "svd_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata → {metadata_file}")


def build_svd_steering_vectors(
    positive_prompts: List[str],
    negative_prompts: List[str],
    layers: List[int],
    model_name: str,
    save_dir: str,
    torch_dtype: torch.dtype = torch.bfloat16
) -> Dict[int, torch.Tensor]:
    """
    Build SVD-based steering vectors from contrastive prompts.

    Args:
        positive_prompts: List of persona-like or trait-positive prompts
        negative_prompts: List of persona-unlike or trait-negative prompts
        layers: List of layer indices to build vectors for
        model_name: HuggingFace model name
        save_dir: Directory to save vectors to
        torch_dtype: Model dtype (default: bfloat16)

    Returns:
        Dictionary mapping layer index to steering vector
    """
    builder = SVDSteeringBuilder(model_name, layers, torch_dtype=torch_dtype)

    # Collect activations
    builder.collect_activations(positive_prompts, negative_prompts)

    # Compute SVD directions
    vectors = builder.compute_svd_directions()

    # Save to disk
    builder.save(save_dir)

    return vectors
