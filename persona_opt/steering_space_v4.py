"""
Steering Space v4 - Semantic ↔ Orthogonal Trait Conversion

This module provides functions to convert between:
- Semantic trait space (R1-R5, R8) - human-interpretable
- Z-normalized space - standardized
- Orthogonal O-space - optimized for steering/optimization

The orthogonal basis is computed via PCA with whitening (ZCA).
"""

import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
BASIS_PATH = BASE_DIR / "orthogonal_basis_v4.npy"
SCALER_PATH = BASE_DIR / "trait_v4_scaler.npy"

# Load basis and scaler
def _load_basis():
    """Load orthogonal basis matrix (6x6)"""
    if not BASIS_PATH.exists():
        raise FileNotFoundError(
            f"Orthogonal basis not found at {BASIS_PATH}. "
            "Run notebooks/orthogonalize_trait_v4.ipynb first."
        )
    return np.load(BASIS_PATH)

def _load_scaler():
    """Load mean and scale for Z-score normalization"""
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. "
            "Run notebooks/orthogonalize_trait_v4.ipynb first."
        )
    return np.load(SCALER_PATH, allow_pickle=True).item()

# Global variables (loaded on import)
W = _load_basis()
scaler_data = _load_scaler()
MEAN = scaler_data["mean"]
SCALE = scaler_data["scale"]

# Conversion functions
def semantic_to_z(w):
    """
    Normalize semantic trait vector w (6-dim) to Z-score

    Args:
        w: array-like of shape (6,) with [R1, R2, R3, R4, R5, R8]

    Returns:
        z: normalized array of shape (6,)
    """
    return (np.array(w) - MEAN) / SCALE

def z_to_semantic(z):
    """
    Convert Z-score back to semantic trait space

    Args:
        z: normalized array of shape (6,)

    Returns:
        w: semantic trait array of shape (6,)
    """
    return z * SCALE + MEAN

def semantic_to_orthogonal(w):
    """
    Convert semantic trait vector to orthogonal O-space

    Args:
        w: array-like of shape (6,) with [R1, R2, R3, R4, R5, R8]

    Returns:
        o: orthogonal space vector of shape (6,)
    """
    z = semantic_to_z(w)
    return W @ z

def orthogonal_to_semantic(o):
    """
    Convert orthogonal O-space vector back to semantic traits

    Args:
        o: orthogonal space vector of shape (6,)

    Returns:
        w: semantic trait array of shape (6,) with [R1, R2, R3, R4, R5, R8]
    """
    z = np.linalg.solve(W, o)
    return z_to_semantic(z)

def get_trait_names():
    """Return list of trait names in order"""
    return ["R1", "R2", "R3", "R4", "R5", "R8"]

def get_semantic_descriptions():
    """Return human-readable descriptions of semantic traits"""
    return {
        "R1": "Self-Other Focus (-1=other, +1=self)",
        "R2": "Expressiveness (-1=concise, +1=verbose)",
        "R3": "Assertiveness (-1=passive, +1=directive)",
        "R4": "Planning/Structure (-1=spontaneous, +1=structured)",
        "R5": "Outlook Valence (-1=negative, +1=positive)",
        "R8": "Time Orientation (-1=past, +1=future)"
    }

# Test function
if __name__ == "__main__":
    print("="*80)
    print("STEERING SPACE V4 - TEST")
    print("="*80)

    # Test round-trip conversion
    print("\nTest 1: Semantic → Orthogonal → Semantic")
    w_original = np.array([0.5, 0.5, 0.0, 0.2, 0.4, 0.0])
    print(f"Original semantic: {w_original}")

    o = semantic_to_orthogonal(w_original)
    print(f"Orthogonal O-space: {o}")

    w_reconstructed = orthogonal_to_semantic(o)
    print(f"Reconstructed semantic: {w_reconstructed}")

    error = np.linalg.norm(w_original - w_reconstructed)
    print(f"Reconstruction error: {error:.6f}")

    if error < 1e-10:
        print("✓ Round-trip conversion successful!")
    else:
        print("⚠️  Warning: Round-trip error too large")

    # Test Z-score normalization
    print("\nTest 2: Z-score normalization")
    z = semantic_to_z(w_original)
    print(f"Z-normalized: {z}")
    w_from_z = z_to_semantic(z)
    print(f"Back to semantic: {w_from_z}")

    # Show trait descriptions
    print("\nTrait Descriptions:")
    for name, desc in get_semantic_descriptions().items():
        print(f"  {name}: {desc}")

    print("\n" + "="*80)
    print("All tests completed successfully!")
    print("="*80)
