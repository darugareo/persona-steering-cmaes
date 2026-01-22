"""
Configuration loader for persona steering experiments.
Provides unified access to experiment settings, prompt templates, and reproducibility controls.
"""

import yaml
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import random
import numpy as np
import torch
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """Unified experiment configuration."""

    # Metadata
    name: str = "persona_steering_evaluation"
    version: str = "1.0.0"
    description: str = ""

    # Reproducibility
    seeds: List[int] = field(default_factory=lambda: [1, 2, 3])
    deterministic: bool = True

    # Model
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"

    # Generation
    max_new_tokens: int = 128
    temperature: float = 0.0
    do_sample: bool = False
    top_p: float = 1.0
    top_k: int = 50

    # Steering
    default_layer: int = 20
    default_alpha: float = 2.0
    layer_range: List[int] = field(default_factory=lambda: [20, 21, 22, 23, 24])
    alpha_range: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    traits: List[str] = field(default_factory=lambda: ["R1", "R2", "R3", "R4", "R5"])
    vectors_dir: str = "data/steering_vectors_v2"

    # Evaluation
    primary_judge: str = "gpt-4o-mini"
    secondary_judge: str = "gpt-4o"
    judge_temperature: float = 0.3
    judge_max_tokens: int = 500
    train_ratio: float = 0.7
    num_eval_prompts: int = 10

    # Baselines
    enabled_baselines: List[str] = field(default_factory=lambda: [
        "base", "prompt_persona", "meandiff", "pca",
        "random_search", "grid_search", "proposed"
    ])

    # Output
    base_dir: str = "reports/experiments"
    results_dir: str = "results"
    figures_dir: str = "figures"
    tables_dir: str = "tables"
    logs_dir: str = "logs"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(
            name=data['experiment']['name'],
            version=data['experiment']['version'],
            description=data['experiment']['description'],
            seeds=data['reproducibility']['seeds'],
            deterministic=data['reproducibility']['deterministic'],
            model_name=data['model']['name'],
            device=data['model']['device'],
            torch_dtype=data['model']['torch_dtype'],
            max_new_tokens=data['generation']['max_new_tokens'],
            temperature=data['generation']['temperature'],
            do_sample=data['generation']['do_sample'],
            top_p=data['generation']['top_p'],
            top_k=data['generation']['top_k'],
            default_layer=data['steering']['default_layer'],
            default_alpha=data['steering']['default_alpha'],
            layer_range=data['steering']['layer_range'],
            alpha_range=data['steering']['alpha_range'],
            traits=data['steering']['traits'],
            vectors_dir=data['steering']['vectors_dir'],
            primary_judge=data['evaluation']['judges']['primary'],
            secondary_judge=data['evaluation']['judges']['secondary'],
            judge_temperature=data['evaluation']['judge_temperature'],
            judge_max_tokens=data['evaluation']['judge_max_tokens'],
            train_ratio=data['evaluation']['train_ratio'],
            num_eval_prompts=data['evaluation']['num_eval_prompts'],
            base_dir=data['output']['base_dir'],
            results_dir=data['output']['structure']['results'],
            figures_dir=data['output']['structure']['figures'],
            tables_dir=data['output']['structure']['tables'],
            logs_dir=data['output']['structure']['logs'],
        )

    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get generation parameters as kwargs dict."""
        return {
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'do_sample': self.do_sample,
            'top_p': self.top_p,
            'top_k': self.top_k,
        }


class PromptTemplateManager:
    """Manages prompt templates with variable substitution."""

    def __init__(self, templates_path: str = "config/prompt_templates.yaml"):
        self.templates_path = Path(templates_path)
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Any]:
        """Load templates from YAML."""
        with open(self.templates_path, 'r') as f:
            return yaml.safe_load(f)

    def get_system_prompt(self, prompt_type: str = "default", **kwargs) -> str:
        """Get system prompt with variable substitution."""
        template = self.templates['system_prompts'][prompt_type]
        return template.format(**kwargs)

    def get_user_prompt(self, prompt_type: str = "direct", **kwargs) -> str:
        """Get user prompt with variable substitution."""
        template = self.templates['user_prompts'][prompt_type]
        return template.format(**kwargs)

    def get_judge_prompt(self, judge_type: str = "persona_fit", **kwargs) -> str:
        """Get judge prompt with variable substitution."""
        template = self.templates['judge_prompts'][judge_type]
        return template.format(**kwargs)

    def get_persona_description(self, template_type: str = "full", **kwargs) -> str:
        """Get persona description with variable substitution."""
        template = self.templates['persona_descriptions'][template_type]
        return template.format(**kwargs)

    def get_evaluation_prompts(self, category: str = "social") -> List[str]:
        """Get list of evaluation prompts for a category."""
        return self.templates['evaluation_prompts'][category]


def set_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ExperimentLogger:
    """Structured logging for experiments."""

    def __init__(self, config: ExperimentConfig, seed: int):
        self.config = config
        self.seed = seed
        self.logs_dir = Path(config.base_dir) / config.logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Create log file
        self.log_file = self.logs_dir / f"experiment_seed{seed}.json"
        self.log_data = {
            'config': {
                'name': config.name,
                'version': config.version,
                'seed': seed,
            },
            'results': {},
            'metrics': {},
            'errors': []
        }

    def log_result(self, method: str, evaluation: str, result: Dict[str, Any]):
        """Log evaluation result."""
        if method not in self.log_data['results']:
            self.log_data['results'][method] = {}
        self.log_data['results'][method][evaluation] = result
        self._save()

    def log_metric(self, name: str, value: Any):
        """Log a single metric."""
        self.log_data['metrics'][name] = value
        self._save()

    def log_error(self, error: str):
        """Log an error."""
        self.log_data['errors'].append(error)
        self._save()

    def _save(self):
        """Save log data to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)


def create_output_directories(config: ExperimentConfig):
    """Create standardized output directory structure."""
    base_dir = Path(config.base_dir)

    directories = [
        base_dir / config.results_dir,
        base_dir / config.figures_dir,
        base_dir / config.tables_dir,
        base_dir / config.logs_dir,
        base_dir / "checkpoints",
    ]

    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)

    return {
        'base': base_dir,
        'results': base_dir / config.results_dir,
        'figures': base_dir / config.figures_dir,
        'tables': base_dir / config.tables_dir,
        'logs': base_dir / config.logs_dir,
        'checkpoints': base_dir / "checkpoints",
    }


# Convenience function
def load_experiment_config(config_path: str = "config/experiment_config.yaml") -> ExperimentConfig:
    """Load experiment configuration from YAML."""
    return ExperimentConfig.from_yaml(config_path)
