"""
Utility to load persona lists from YAML configuration.
"""

import yaml
from pathlib import Path
from typing import List


def load_persona_list(config_path: str = "persona-opt/personas.yaml") -> List[str]:
    """
    Load list of persona IDs from YAML config.

    Args:
        config_path: Path to personas.yaml

    Returns:
        List of persona IDs
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Persona list config not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    personas = config.get('personas', [])

    if not personas:
        raise ValueError(f"No personas found in {config_path}")

    return personas


def get_persona_list_from_args(args) -> List[str]:
    """
    Get persona list from command-line arguments.

    Supports two modes:
    1. --persona-id <single_id>: Single persona
    2. --persona-list <yaml_path>: Multiple personas from YAML

    Args:
        args: argparse.Namespace with persona_id and/or persona_list

    Returns:
        List of persona IDs
    """
    if hasattr(args, 'persona_list') and args.persona_list:
        # Load from YAML
        return load_persona_list(args.persona_list)
    elif hasattr(args, 'persona_id') and args.persona_id:
        # Single persona
        return [args.persona_id]
    else:
        # Default: load from default config
        return load_persona_list()
