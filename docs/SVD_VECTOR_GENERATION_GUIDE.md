# SVD Steering Vector Generation Guide

**Complete instructions for generating SVD-based steering vectors for 5-trait persona optimization.**

---

## Overview

This guide provides step-by-step instructions to:
1. Extract POS/NEG prompts from trait_prompts_v2.json
2. Generate SVD steering vectors for R1-R5 traits
3. Produce 25 vector files (5 traits × 5 layers) ready for CMA-ES optimization

**Input**: `/data01/nakata/master_thesis/persona2/data/prompts/trait_prompts_v2.json`

**Output**: `/data01/nakata/master_thesis/persona2/data/steering_vectors_v2/{R1-R5}/layer{20-24}_svd.pt`

**Model**: `meta-llama/Meta-Llama-3-8B-Instruct`

**Layers**: 20, 21, 22, 23, 24

---

## Prerequisites

✅ `persona_opt/svd_vector_builder.py` (already implemented)
✅ `scripts/run_build_svd_vectors.py` (already implemented)
✅ `data/prompts/trait_prompts_v2.json` (already generated)

---

## Step 1: Extract POS/NEG Prompts for Each Trait

Create a helper script to convert trait_prompts_v2.json into separate POS/NEG JSON files for each trait.

**Script to create**: `scripts/extract_trait_prompts.py`

```python
"""
Extract POS/NEG prompt lists from trait_prompts_v2.json for SVD vector building.
"""

import json
from pathlib import Path

def extract_trait_prompts(input_file: str, output_dir: str):
    """
    Extract POS/NEG prompts for each trait (R1-R5).

    Args:
        input_file: Path to trait_prompts_v2.json
        output_dir: Directory to save extracted prompts
    """
    # Load trait prompts
    with open(input_file, 'r', encoding='utf-8') as f:
        trait_data = json.load(f)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract for each trait
    for trait_name, pairs in trait_data.items():
        pos_prompts = [pair["pos"] for pair in pairs]
        neg_prompts = [pair["neg"] for pair in pairs]

        # Save POS prompts
        pos_file = output_path / f"{trait_name}_positive.json"
        with open(pos_file, 'w', encoding='utf-8') as f:
            json.dump(pos_prompts, f, indent=2, ensure_ascii=False)

        # Save NEG prompts
        neg_file = output_path / f"{trait_name}_negative.json"
        with open(neg_file, 'w', encoding='utf-8') as f:
            json.dump(neg_prompts, f, indent=2, ensure_ascii=False)

        print(f"✓ {trait_name}: {len(pos_prompts)} POS, {len(neg_prompts)} NEG")
        print(f"  → {pos_file}")
        print(f"  → {neg_file}")

if __name__ == "__main__":
    extract_trait_prompts(
        input_file="data/prompts/trait_prompts_v2.json",
        output_dir="data/prompts/extracted"
    )
    print("\n✅ All trait prompts extracted successfully!")
```

**Execute**:
```bash
python scripts/extract_trait_prompts.py
```

**Expected output**:
```
data/prompts/extracted/R1_positive.json
data/prompts/extracted/R1_negative.json
data/prompts/extracted/R2_positive.json
data/prompts/extracted/R2_negative.json
...
data/prompts/extracted/R5_positive.json
data/prompts/extracted/R5_negative.json
```

---

## Step 2: Generate SVD Vectors for Each Trait

Run SVD vector generation for R1-R5.

### Option A: Run All Traits (Bash Loop)

**Create**: `scripts/run_all_svd_extraction.sh`

```bash
#!/bin/bash

# SVD Vector Generation for All Traits
# Run this to generate all 25 steering vectors (5 traits × 5 layers)

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
LAYERS="20,21,22,23,24"
BASE_DIR="data"

echo "=========================================="
echo "SVD Steering Vector Generation"
echo "=========================================="
echo "Model: $MODEL"
echo "Layers: $LAYERS"
echo ""

for TRAIT in R1 R2 R3 R4 R5; do
    echo "----------------------------------------"
    echo "Processing trait: $TRAIT"
    echo "----------------------------------------"

    python scripts/run_build_svd_vectors.py \
        --positive ${BASE_DIR}/prompts/extracted/${TRAIT}_positive.json \
        --negative ${BASE_DIR}/prompts/extracted/${TRAIT}_negative.json \
        --layers ${LAYERS} \
        --model ${MODEL} \
        --save_dir ${BASE_DIR}/steering_vectors_v2/${TRAIT}/ \
        --dtype bfloat16

    if [ $? -eq 0 ]; then
        echo "✅ $TRAIT completed successfully"
    else
        echo "❌ $TRAIT failed"
        exit 1
    fi
    echo ""
done

echo "=========================================="
echo "✅ All SVD vectors generated successfully!"
echo "=========================================="
echo ""
echo "Output location:"
echo "  ${BASE_DIR}/steering_vectors_v2/"
echo ""
echo "Generated files:"
ls -lh ${BASE_DIR}/steering_vectors_v2/*/layer*.pt
```

**Execute**:
```bash
chmod +x scripts/run_all_svd_extraction.sh
./scripts/run_all_svd_extraction.sh
```

### Option B: Run Individual Traits

**R1: Self-Other Focus**
```bash
python scripts/run_build_svd_vectors.py \
  --positive data/prompts/extracted/R1_positive.json \
  --negative data/prompts/extracted/R1_negative.json \
  --layers 20,21,22,23,24 \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --save_dir data/steering_vectors_v2/R1/
```

**R2: Formality**
```bash
python scripts/run_build_svd_vectors.py \
  --positive data/prompts/extracted/R2_positive.json \
  --negative data/prompts/extracted/R2_negative.json \
  --layers 20,21,22,23,24 \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --save_dir data/steering_vectors_v2/R2/
```

**R3: Relationship Distance**
```bash
python scripts/run_build_svd_vectors.py \
  --positive data/prompts/extracted/R3_positive.json \
  --negative data/prompts/extracted/R3_negative.json \
  --layers 20,21,22,23,24 \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --save_dir data/steering_vectors_v2/R3/
```

**R4: Narrative Mode**
```bash
python scripts/run_build_svd_vectors.py \
  --positive data/prompts/extracted/R4_positive.json \
  --negative data/prompts/extracted/R4_negative.json \
  --layers 20,21,22,23,24 \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --save_dir data/steering_vectors_v2/R4/
```

**R5: Emotional Tone**
```bash
python scripts/run_build_svd_vectors.py \
  --positive data/prompts/extracted/R5_positive.json \
  --negative data/prompts/extracted/R5_negative.json \
  --layers 20,21,22,23,24 \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --save_dir data/steering_vectors_v2/R5/
```

---

## Step 3: Verify Generated Vectors

**Expected file structure**:
```
data/steering_vectors_v2/
├── R1/
│   ├── layer20_svd.pt
│   ├── layer21_svd.pt
│   ├── layer22_svd.pt
│   ├── layer23_svd.pt
│   ├── layer24_svd.pt
│   └── svd_metadata.json
├── R2/
│   ├── layer20_svd.pt
│   ...
├── R3/
├── R4/
└── R5/
```

**Total files**: 25 `.pt` files + 5 `svd_metadata.json` files

**Verification script**: `scripts/verify_svd_vectors.py`

```python
"""
Verify that all SVD vectors have been generated correctly.
"""

import torch
from pathlib import Path
import json

def verify_svd_vectors(base_dir: str = "data/steering_vectors_v2"):
    """Verify all SVD vectors are present and valid."""

    base_path = Path(base_dir)
    traits = ["R1", "R2", "R3", "R4", "R5"]
    layers = [20, 21, 22, 23, 24]

    print("=" * 60)
    print("SVD Vector Verification")
    print("=" * 60)

    all_valid = True

    for trait in traits:
        trait_dir = base_path / trait
        print(f"\n{trait}:")

        if not trait_dir.exists():
            print(f"  ❌ Directory missing: {trait_dir}")
            all_valid = False
            continue

        # Check metadata
        metadata_file = trait_dir / "svd_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"  ✓ Metadata: {metadata['num_positive']} POS, {metadata['num_negative']} NEG")
        else:
            print(f"  ⚠️  Metadata missing")

        # Check vector files
        for layer in layers:
            vector_file = trait_dir / f"layer{layer}_svd.pt"

            if not vector_file.exists():
                print(f"  ❌ Missing: layer{layer}_svd.pt")
                all_valid = False
            else:
                vector = torch.load(vector_file)
                norm = vector.norm().item()
                print(f"  ✓ layer{layer}_svd.pt: shape={vector.shape}, norm={norm:.6f}")

                if abs(norm - 1.0) > 0.01:
                    print(f"    ⚠️  Warning: Vector not normalized (norm={norm})")

    print("\n" + "=" * 60)
    if all_valid:
        print("✅ All SVD vectors verified successfully!")
    else:
        print("❌ Some vectors are missing or invalid")
    print("=" * 60)

if __name__ == "__main__":
    verify_svd_vectors()
```

**Execute**:
```bash
python scripts/verify_svd_vectors.py
```

---

## Step 4: Integration with Internal Steering

The generated vectors will be used in `persona_opt/internal_steering_l3.py` for multi-trait steering:

```python
# Load SVD vectors for all traits
trait_vectors = {}
for trait in ["R1", "R2", "R3", "R4", "R5"]:
    vector_path = f"data/steering_vectors_v2/{trait}/layer{layer}_svd.pt"
    trait_vectors[trait] = torch.load(vector_path)

# Combine with CMA-ES weights
steering_vec = sum(
    weights[trait] * trait_vectors[trait]
    for trait in ["R1", "R2", "R3", "R4", "R5"]
)

# Apply steering
steerer.register_hooks(steering_vector=steering_vec, alpha=alpha)
```

---

## Summary

**✅ Prerequisites**:
- SVD builder implemented
- Trait prompts generated (125 POS/NEG pairs)

**✅ Execution steps**:
1. Extract trait prompts → separate JSON files
2. Run SVD generation for R1-R5
3. Verify 25 vector files generated
4. Ready for CMA-ES optimization

**✅ Output**:
- 25 steering vectors (5 traits × 5 layers)
- Metadata files for reproducibility
- Ready for persona-specific weight optimization

**Next phase**: CMA-ES optimization to find optimal trait weights per persona

---

**Status**: Ready for execution

**Estimated time**: ~30-60 minutes (depends on GPU availability)

**Required resources**: GPU with 16GB+ VRAM for Llama-3-8B-Instruct
