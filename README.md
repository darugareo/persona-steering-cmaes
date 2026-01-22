# Persona-based Language Model Steering via CMA-ES Optimization

This repository implements a persona-based language model steering framework using CMA-ES optimization and SVD-based steering vectors. The system enables large language models to reproduce specific conversational personas through internal activation steering.

## Overview

The framework consists of three main components:

1. **SVD-based Steering Vectors**: Extract trait-specific steering directions from contrastive activations
2. **CMA-ES Optimization**: Optimize trait weight combinations to reproduce target personas
3. **Persona-aware Evaluation**: Automatic evaluation using LLM-as-a-judge with persona profiles

## Project Structure

```
persona-steering-cmaes/
├── persona_opt/          # Core optimization and steering modules
│   ├── cmaes_persona_optimizer.py    # CMA-ES optimizer
│   ├── svd_vector_builder.py         # SVD steering vector construction
│   ├── internal_steering_l3.py       # Llama-3 activation steering
│   ├── persona_judge_evaluator.py    # Persona-aware judge
│   ├── evaluator.py                  # General evaluator
│   ├── judge.py                      # Judge implementation
│   ├── baselines/                    # Baseline methods
│   │   ├── proposed.py               # Proposed SVD+CMA-ES method
│   │   ├── meandiff.py               # Mean difference baseline
│   │   ├── pca_steering.py           # PCA-based steering
│   │   ├── prompt_persona.py         # Prompt-based persona
│   │   └── random_search.py          # Random search baseline
│   ├── evaluation/                   # Advanced evaluation modules
│   │   ├── multi_judge.py            # Multi-judge evaluation
│   │   ├── cross_layer.py            # Cross-layer analysis
│   │   └── human_eval.py             # Human evaluation support
│   └── utils/                        # Utility functions
├── persona_judge/        # Persona profile and judge prompt generation
│   ├── persona_profile.py            # Profile generation from conversations
│   ├── feature_extractor.py          # Feature extraction
│   ├── sample_selector.py            # Representative sample selection
│   ├── judge_prompt_builder.py       # Judge prompt construction
│   └── conversation_loader.py        # Load and parse conversations
├── scripts/              # Executable scripts (150+ scripts)
│   ├── run_build_svd_vectors.py      # Build SVD steering vectors
│   ├── run_persona_optimization.py   # Single persona CMA-ES optimization
│   ├── run_7personas_optimization.py # Multi-persona optimization
│   ├── run_baseline_comparison.py    # Compare baselines
│   ├── generate_phase1_report.py     # Phase 1 evaluation reports
│   ├── evaluate_all_conditions.py    # Comprehensive evaluation
│   ├── statistical_analysis.py       # Statistical significance tests
│   └── ...                           # Many more analysis scripts
├── personas/             # Persona configurations (35 personas)
│   ├── episode-184019_A/             # Example persona
│   │   ├── persona_profile.txt       # Natural language profile
│   │   ├── persona_features.json     # Extracted features
│   │   ├── persona_samples.json      # Representative samples
│   │   ├── final_judge_prompt.txt    # Judge evaluation prompt
│   │   └── raw_conversations.json    # Source conversations
│   └── ...
├── personas_cc/          # ConvAI2 persona train/test splits
├── persona_profiles/     # Pre-generated persona profiles (JSON)
├── data/                 # Shared data and configurations
│   ├── all_persona_profiles.json     # All persona profiles
│   ├── personas_final_10.txt         # Selected 10 personas
│   └── report_data_7personas.json    # 7 personas experiment data
├── docs/                 # Documentation and guides
│   ├── CMAES_OPTIMIZATION_GUIDE.md   # Optimization guide
│   ├── EVALUATION_GUIDE.md           # Evaluation instructions
│   ├── LAMP_EXPERIMENTAL_DESIGN.md   # LaMP experiments
│   └── ...                           # Implementation summaries
├── experiments/          # Experiment-specific code
│   ├── base_vs_steering_analysis/    # Baseline vs steering analysis
│   ├── adaptive_trait_selection/     # Adaptive trait methods
│   └── two_trait_steering/           # 2-trait experiments
├── notebooks/            # Jupyter notebooks for analysis
├── SETUP_GUIDE.md        # Comprehensive setup instructions
├── LICENSE               # MIT License
└── README.md             # This file

```

## Setup

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd persona-steering-cmaes

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. HuggingFace Model Access

The framework uses `meta-llama/Meta-Llama-3-8B-Instruct`. You need to:

#### Step 1: Get HuggingFace Access Token
1. Create account at [huggingface.co](https://huggingface.co)
2. Request access to [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
3. Generate access token at [Settings > Access Tokens](https://huggingface.co/settings/tokens)

#### Step 2: Login to HuggingFace
```bash
# Install HuggingFace CLI (included in requirements.txt)
huggingface-cli login

# Or set environment variable
export HF_TOKEN=your_huggingface_token
```

#### Step 3: Verify Model Access
```bash
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')"
```

The model will be automatically downloaded to `~/.cache/huggingface/` on first use (~16GB).

**GPU Requirements:**
- Minimum: 16GB VRAM (for 8B model)
- Recommended: 24GB+ VRAM
- CPU inference is possible but very slow

### 3. OpenAI API Configuration

For evaluation using LLM-as-a-judge (GPT-4):

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env  # or vim, code, etc.
```

Add to `.env`:
```bash
OPENAI_API_KEY=sk-your-api-key-here

# Optional: Configure judge model (default: gpt-4o-mini)
JUDGE_MODEL=gpt-4o-mini

# Optional: Configure generator model (for baselines)
GENERATOR_MODEL=gpt-4o
```

**Get OpenAI API Key:**
1. Sign up at [platform.openai.com](https://platform.openai.com)
2. Go to [API Keys](https://platform.openai.com/api-keys)
3. Create new secret key
4. Add billing information (pay-as-you-go)

**Estimated API Costs:**
- Single persona optimization (~50 iterations, 10 prompts): $0.50-$2.00
- Full evaluation (13 personas): $10-$30
- Uses `gpt-4o-mini` by default (cheaper than GPT-4)

## Quick Start

### 1. Build SVD Steering Vectors

First, generate trait-specific steering vectors (R1-R5):

```bash
python3 scripts/run_build_svd_vectors.py
```

This creates steering vectors in `data/steering_vectors_v2/`.

### 2. Run Persona Optimization

Optimize trait weights for a specific persona:

```bash
python3 scripts/run_persona_optimization.py \
    --persona_id episode-184019_A \
    --layer 20 \
    --alpha 2.0 \
    --iterations 50
```

### 3. Evaluate with Baselines

Compare the proposed method against baselines:

```bash
python3 scripts/run_baseline_comparison.py \
    --persona_id episode-184019_A \
    --methods proposed meandiff pca prompt
```

### 4. Generate Evaluation Report

Create comprehensive evaluation reports:

```bash
python3 scripts/generate_phase1_report.py \
    --persona_id episode-184019_A
```

## Available Personas

The repository includes 35 pre-configured personas from conversational datasets. Example personas include:

- episode-184019_A
- episode-239427_A
- episode-118328_B
- episode-225888_A
- episode-134226_A
- episode-179307_A
- episode-137872_B
- episode-158821_B
- ...and 27 more

Each persona directory contains:
- `persona_profile.txt` - Natural language profile
- `persona_features.json` - Extracted features
- `persona_samples.json` - Representative samples
- `final_judge_prompt.txt` - Judge evaluation prompt
- `raw_conversations.json` - Source conversations

## Key Scripts

### Optimization

- `run_persona_optimization.py` - Single persona CMA-ES optimization
- `run_7personas_optimization.py` - Multi-persona batch optimization
- `run_multi_persona_optimization.py` - Advanced multi-persona optimization

### Evaluation

- `run_baseline_comparison.py` - Compare methods
- `run_multi_judge_eval.py` - Multi-judge evaluation
- `run_lamp_evaluation.py` - LaMP dataset evaluation
- `run_truthfulqa_eval.py` - TruthfulQA evaluation
- `run_mmlu_eval.py` - MMLU evaluation

### Analysis

- `generate_phase1_report.py` - Phase 1 evaluation report
- `analyze_optimization_results.py` - Analyze CMA-ES convergence
- `statistical_analysis.py` - Statistical significance tests

## Baseline Methods

The framework includes several baseline methods for comparison:

- **Proposed**: SVD + CMA-ES optimization (this work)
- **MeanDiff**: Simple mean difference steering
- **PCA**: PCA-based steering vectors
- **Prompt**: Prompt-based persona (no internal steering)
- **Random**: Random search over trait weights

## Configuration

### Experiment Configuration

Edit `config/experiment_config.yaml` to configure:
- Model parameters
- Optimization settings
- Evaluation prompts
- Generation parameters

### Prompt Templates

Customize evaluation prompts in `config/prompt_templates.yaml`.

## Output Structure

Results are organized as follows:

```
results/
├── same_model/           # Same model evaluations
├── cross_model/          # Cross-model evaluations
├── judge_evaluation/     # Judge evaluation logs
└── lamp7/               # LaMP-7 results
```

## Development

### Adding a New Persona

1. Create directory in `personas/`:
```bash
mkdir personas/new_persona_id
```

2. Add required files:
- `raw_conversations.json`
- `persona_profile.txt`
- `persona_features.json`
- `persona_samples.json`

3. Generate judge prompt:
```bash
python3 scripts/main_generate_judge.py --persona_id new_persona_id
```

### Adding a New Baseline

1. Create a new file in `persona_opt/baselines/`
2. Inherit from `BasePersonaMethod` in `baselines/base.py`
3. Implement required methods

## Citation

If you use this code, please cite:

```bibtex
@article{your_paper,
  title={Persona-based Language Model Steering via CMA-ES Optimization},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Taisei Nakata

## Acknowledgments

This work uses:
- Meta's Llama-3-8B-Instruct model
- OpenAI API for evaluation
- CMA-ES optimization library
