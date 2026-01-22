"""
Evaluation prompt management for persona-aware steering.

Provides utilities for loading and filtering evaluation prompts.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional


def load_eval_prompts(
    prompts_file: Path = Path("data/eval_prompts/persona_eval_prompts_v1.json"),
    category: Optional[str] = None,
    num_prompts: Optional[int] = None
) -> List[str]:
    """
    Load evaluation prompts from JSON file.

    Args:
        prompts_file: Path to prompts JSON file
        category: If specified, only load prompts from this category
        num_prompts: If specified, limit to first N prompts

    Returns:
        List of prompt strings
    """
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    with open(prompts_file) as f:
        data = json.load(f)

    prompts = []
    for prompt_obj in data["prompts"]:
        # Filter by category if specified
        if category is not None and prompt_obj["category"] != category:
            continue

        prompts.append(prompt_obj["text"])

        # Limit number if specified
        if num_prompts is not None and len(prompts) >= num_prompts:
            break

    return prompts


def load_eval_prompts_with_metadata(
    prompts_file: Path = Path("data/eval_prompts/persona_eval_prompts_v1.json"),
    category: Optional[str] = None,
    num_prompts: Optional[int] = None
) -> List[Dict]:
    """
    Load evaluation prompts with full metadata.

    Args:
        prompts_file: Path to prompts JSON file
        category: If specified, only load prompts from this category
        num_prompts: If specified, limit to first N prompts

    Returns:
        List of prompt dictionaries with metadata
    """
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    with open(prompts_file) as f:
        data = json.load(f)

    prompts = []
    for prompt_obj in data["prompts"]:
        # Filter by category if specified
        if category is not None and prompt_obj["category"] != category:
            continue

        prompts.append(prompt_obj)

        # Limit number if specified
        if num_prompts is not None and len(prompts) >= num_prompts:
            break

    return prompts


def get_prompt_categories(
    prompts_file: Path = Path("data/eval_prompts/persona_eval_prompts_v1.json")
) -> List[str]:
    """
    Get list of available prompt categories.

    Args:
        prompts_file: Path to prompts JSON file

    Returns:
        List of category names
    """
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    with open(prompts_file) as f:
        data = json.load(f)

    return list(data["categories"].keys())


def get_prompts_summary(
    prompts_file: Path = Path("data/eval_prompts/persona_eval_prompts_v1.json")
) -> Dict:
    """
    Get summary statistics about prompt set.

    Args:
        prompts_file: Path to prompts JSON file

    Returns:
        Dictionary with summary stats
    """
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    with open(prompts_file) as f:
        data = json.load(f)

    return {
        "version": data.get("version", "unknown"),
        "total_prompts": len(data["prompts"]),
        "categories": data.get("categories", {}),
        "description": data.get("description", "")
    }


# Example usage
if __name__ == "__main__":
    # Print summary
    summary = get_prompts_summary()
    print(f"Prompt Set Version: {summary['version']}")
    print(f"Total Prompts: {summary['total_prompts']}")
    print(f"Description: {summary['description']}\n")

    print("Categories:")
    for category, count in summary['categories'].items():
        print(f"  {category}: {count} prompts")

    # Load all prompts
    print("\n--- All Prompts ---")
    prompts = load_eval_prompts()
    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. {prompt}")

    # Load specific category
    print("\n--- Social Support Category Only ---")
    social_prompts = load_eval_prompts(category="social_support")
    for i, prompt in enumerate(social_prompts, 1):
        print(f"{i}. {prompt}")

    # Load limited number
    print("\n--- First 5 Prompts ---")
    limited_prompts = load_eval_prompts(num_prompts=5)
    for i, prompt in enumerate(limited_prompts, 1):
        print(f"{i}. {prompt}")
