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
