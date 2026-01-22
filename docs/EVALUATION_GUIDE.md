yosi# Persona Steering Evaluation Guide

Complete guide for evaluating optimized persona steering vectors.

**Version**: 1.0
**Date**: 2025-12-08
**Target**: Researchers validating persona reproduction quality

---

## Table of Contents

1. [Overview](#overview)
2. [Evaluation Tasks](#evaluation-tasks)
   - [1. Train/Test Split](#1-traintestsplit-evaluation)
   - [2. Cross-Layer Transfer](#2-cross-layer-transfer-evaluation)
   - [3. Alpha Sensitivity](#3-alpha-sensitivity-evaluation)
   - [4. Multi-Turn Stability](#4-multi-turn-stability-evaluation)
   - [5. Multi-Judge Reliability](#5-multi-judge-reliability-evaluation)
   - [6. Human Evaluation](#6-human-evaluation)
3. [Quick Start](#quick-start)
4. [Interpreting Results](#interpreting-results)
5. [Best Practices](#best-practices)
6. [Using Results in Papers](#using-results-in-papers)

---

## Overview

### Purpose

This evaluation framework validates that CMA-ES optimized persona steering vectors **truly reproduce target personas** across multiple dimensions:

- **Generalization**: Does it work on unseen prompts?
- **Transferability**: Does it work across different layers?
- **Robustness**: Does it work with different steering strengths?
- **Consistency**: Does it maintain persona across conversations?
- **Reliability**: Do multiple judges agree?
- **Human validity**: Do humans perceive persona fit?

### Architecture

```
persona_opt/evaluation/
├── utils.py              # Shared utilities
├── train_test.py         # Generalization evaluation
├── cross_layer.py        # Layer transferability
├── alpha_sensitivity.py  # Steering strength robustness
├── multi_turn.py         # Conversation consistency
├── multi_judge.py        # Inter-judge agreement
└── human_eval.py         # Human study generator

scripts/
├── run_train_test.py
├── run_cross_layer.py
├── run_alpha_sensitivity.py
├── run_multi_turn.py
├── run_multi_judge.py
└── run_human_eval_template.py

reports/evaluation/
└── {task_name}/{persona_id}/
    ├── result.json       # Full results
    ├── score_summary.md  # Human-readable summary
    └── plot.png          # Visualization
```

---

## Evaluation Tasks

### 1. Train/Test Split Evaluation

**Purpose**: Detect overfitting to optimization prompts

**What it does**:
- Splits 10 prompts into 7 train / 3 test
- Evaluates persona fit on both sets
- Compares generalization gap

**Usage**:

```bash
python scripts/run_train_test.py \
  --persona-id episode-184019_A \
  --prompts-file data/prompts/persona_eval_prompts.json \
  --num-prompts 10 \
  --train-ratio 0.7 \
  --output-dir reports/evaluation/train_test/episode-184019_A
```

**Key Parameters**:
- `--train-ratio`: Proportion of training data (default: 0.7)
- `--seed`: Random seed for reproducibility (default: 42)
- `--judge-model`: Which LLM judge to use (default: gpt-4o-mini)

**Output**:
- `result.json`: Train/test scores and statistics
- `plot.png`: Bar chart comparing train vs test
- `score_summary.md`: Interpretation

**Example Output**:
```
Train score:  5.00
Test score:   4.67
Generalization gap: 0.33
Overfitting: No
```

**Interpretation**:
- **Gap < 0.3**: Excellent generalization
- **Gap 0.3-0.5**: Good generalization
- **Gap > 0.5**: Possible overfitting

---

### 2. Cross-Layer Transfer Evaluation

**Purpose**: Test if weights transfer across layers

**What it does**:
- Uses optimized weights (w1-w5, α)
- Applies to layers 20, 21, 22, 23, 24
- Measures persona fit for each layer

**Usage**:

```bash
python scripts/run_cross_layer.py \
  --persona-id episode-184019_A \
  --prompts-file data/prompts/persona_eval_prompts.json \
  --layers 20 21 22 23 24 \
  --output-dir reports/evaluation/cross_layer/episode-184019_A
```

**Key Parameters**:
- `--layers`: Which layers to test (default: 20-24)
- `--num-prompts`: Number of prompts (default: 10)

**Output**:
- Line plot showing score vs layer
- Win rate vs layer
- Best performing layer identification

**Example Output**:
```
Optimized layer: 20
Optimized score: 5.00
Best layer: 21
Best score: 4.90
Transferable: Limited
```

**Interpretation**:
- **Best layer = optimized layer**: Weights are layer-specific
- **Best layer ± 1**: Moderate transferability
- **Best layer far from optimized**: High transferability (rare)

**Value for papers**:
- Shows where persona information resides in the model
- Validates that optimization didn't just "memorize" layer 20

---

### 3. Alpha Sensitivity Evaluation

**Purpose**: Test robustness to steering strength

**What it does**:
- Sweeps α from 0.5 to 3.0
- Measures persona fit at each α
- Detects optimal steering strength range

**Usage**:

```bash
python scripts/run_alpha_sensitivity.py \
  --persona-id episode-184019_A \
  --prompts-file data/prompts/persona_eval_prompts.json \
  --alpha-values 0.5 1.0 1.5 2.0 2.5 3.0 \
  --output-dir reports/evaluation/alpha_sensitivity/episode-184019_A
```

**Key Parameters**:
- `--alpha-values`: List of α values to test

**Output**:
- Curve: Persona fit vs α
- Curve: Win rate vs α
- Optimal α range identification

**Example Output**:
```
Optimized alpha: 2.0
Optimized score: 5.00
Best alpha: 2.0
Best score: 5.00
Score range: 0.35
Robust: Yes
```

**Interpretation**:
- **Range < 0.5**: Robust to α variation
- **Range 0.5-1.0**: Moderately sensitive
- **Range > 1.0**: Highly sensitive (tune carefully)

**Value for papers**:
- Shows steering is not fragile
- Identifies safe α range for deployment

---

### 4. Multi-Turn Stability Evaluation

**Purpose**: Detect persona drift in conversations

**What it does**:
- Runs 5-turn conversations with steering
- Evaluates persona fit at each turn
- Detects degradation over time

**Usage**:

```bash
python scripts/run_multi_turn.py \
  --persona-id episode-184019_A \
  --prompts-file data/prompts/persona_eval_prompts.json \
  --num-conversations 5 \
  --num-turns 5 \
  --output-dir reports/evaluation/multi_turn/episode-184019_A
```

**Key Parameters**:
- `--num-conversations`: Number of dialogues to test
- `--num-turns`: Turns per conversation

**Output**:
- Plot: Score vs turn number
- Drift analysis: First turn vs last turn
- Box plots showing score distribution

**Example Output**:
```
Number of turns: 5
First turn score: 4.80
Last turn score: 4.65
Drift: -0.15
Stable: Yes
```

**Interpretation**:
- **|Drift| < 0.5**: Stable persona
- **|Drift| 0.5-1.0**: Moderate drift
- **|Drift| > 1.0**: Significant degradation

**Value for papers**:
- Most practical evaluation (mimics real usage)
- Shows persona is maintained in context

---

### 5. Multi-Judge Reliability Evaluation

**Purpose**: Validate evaluation is not judge-specific

**What it does**:
- Evaluates same responses with multiple judges
- Computes inter-judge agreement (Spearman ρ)
- Identifies most lenient/strict judges

**Usage**:

```bash
python scripts/run_multi_judge.py \
  --persona-id episode-184019_A \
  --prompts-file data/prompts/persona_eval_prompts.json \
  --judges gpt-4o-mini gpt-4o claude-3-5-sonnet-20241022 \
  --output-dir reports/evaluation/multi_judge/episode-184019_A
```

**Key Parameters**:
- `--judges`: List of LLM judge models
- Supported: `gpt-4o-mini`, `gpt-4o`, `claude-3-5-sonnet-20241022`

**Output**:
- Mean scores per judge
- Win rates per judge
- Correlation heatmap
- Pairwise Spearman correlations

**Example Output**:
```
Number of judges: 3
Mean agreement (Spearman): 0.85
Most lenient: gpt-4o-mini
Most strict: claude-3-5-sonnet-20241022
Reliable: Yes
```

**Interpretation**:
- **ρ > 0.7**: High agreement (reliable)
- **ρ 0.5-0.7**: Moderate agreement
- **ρ < 0.5**: Low agreement (investigate)

**Value for papers**:
- Essential for validating automated evaluation
- Shows results are not artifacts of one judge

---

### 6. Human Evaluation

**Purpose**: Ground truth validation

**What it does**:
- Generates randomized baseline/steered/persona samples
- Exports CSV for human raters
- Provides rating instructions

**Usage**:

```bash
python scripts/run_human_eval_template.py \
  --persona-id episode-184019_A \
  --prompts-file data/prompts/persona_eval_prompts.json \
  --num-samples 20 \
  --include-persona-sample \
  --output-dir reports/evaluation/human_eval/episode-184019_A
```

**Key Parameters**:
- `--num-samples`: Number of comparison items
- `--include-persona-sample`: 3-way (baseline/steered/real) vs 2-way
- `--seed`: For reproducible randomization

**Output Files**:

1. **human_evaluation.csv**: For raters (no labels)
   ```csv
   item_id,prompt,response_A,response_B,response_C,rating_A,rating_B,rating_C,best_response,notes
   ```

2. **answer_key.csv**: True labels (keep private)
   ```csv
   item_id,prompt,response_A,response_B,response_C,label_A,label_B,label_C
   ```

3. **instructions.md**: Evaluation guide for humans

4. **evaluation_data.json**: Full data with metadata

**Workflow**:

1. Generate data with script
2. Share `human_evaluation.csv` and `instructions.md` with raters
3. Collect completed CSVs
4. Compute inter-rater agreement (Fleiss' κ)
5. Compare human ratings to LLM judge ratings

**Value for papers**:
- **Required** for publication in top venues
- Shows persona fit is human-perceptible
- Validates automated evaluation

---

## Quick Start

### Running All Evaluations

```bash
# Set variables
PERSONA_ID="episode-184019_A"
PROMPTS="data/prompts/persona_eval_prompts.json"

# 1. Train/Test Split
python scripts/run_train_test.py \
  --persona-id $PERSONA_ID \
  --prompts-file $PROMPTS

# 2. Cross-Layer Transfer
python scripts/run_cross_layer.py \
  --persona-id $PERSONA_ID \
  --prompts-file $PROMPTS

# 3. Alpha Sensitivity
python scripts/run_alpha_sensitivity.py \
  --persona-id $PERSONA_ID \
  --prompts-file $PROMPTS

# 4. Multi-Turn Stability
python scripts/run_multi_turn.py \
  --persona-id $PERSONA_ID \
  --prompts-file $PROMPTS

# 5. Multi-Judge Reliability
python scripts/run_multi_judge.py \
  --persona-id $PERSONA_ID \
  --prompts-file $PROMPTS

# 6. Human Evaluation Data
python scripts/run_human_eval_template.py \
  --persona-id $PERSONA_ID \
  --prompts-file $PROMPTS \
  --include-persona-sample
```

### Batch Script

Create `run_all_evaluations.sh`:

```bash
#!/bin/bash
PERSONA_ID=$1
PROMPTS="data/prompts/persona_eval_prompts.json"

echo "Running all evaluations for $PERSONA_ID..."

python scripts/run_train_test.py --persona-id $PERSONA_ID --prompts-file $PROMPTS
python scripts/run_cross_layer.py --persona-id $PERSONA_ID --prompts-file $PROMPTS
python scripts/run_alpha_sensitivity.py --persona-id $PERSONA_ID --prompts-file $PROMPTS
python scripts/run_multi_turn.py --persona-id $PERSONA_ID --prompts-file $PROMPTS
python scripts/run_multi_judge.py --persona-id $PERSONA_ID --prompts-file $PROMPTS
python scripts/run_human_eval_template.py --persona-id $PERSONA_ID --prompts-file $PROMPTS --include-persona-sample

echo "All evaluations complete!"
echo "Results in: reports/evaluation/*/$PERSONA_ID/"
```

Usage:
```bash
chmod +x run_all_evaluations.sh
./run_all_evaluations.sh episode-184019_A
```

---

## Interpreting Results

### Success Criteria

| Evaluation | Excellent | Good | Concerning |
|------------|-----------|------|------------|
| **Train/Test Gap** | < 0.3 | 0.3-0.5 | > 0.5 |
| **Cross-Layer Best** | Same layer | ±1 layer | >±2 layers |
| **Alpha Range** | < 0.5 | 0.5-1.0 | > 1.0 |
| **Multi-Turn Drift** | < 0.3 | 0.3-0.5 | > 0.5 |
| **Judge Agreement** | ρ > 0.7 | ρ 0.5-0.7 | ρ < 0.5 |
| **Human Agreement** | κ > 0.6 | κ 0.4-0.6 | κ < 0.4 |

### Red Flags

🚩 **Overfitting**: Train score 5.0, test score < 4.0
→ *Action*: Increase training prompts, reduce CMA-ES iterations

🚩 **Layer-specific**: Best layer far from optimized layer
→ *Action*: Re-optimize or investigate trait vector quality

🚩 **Fragile steering**: Score drops >1.0 when α changes ±0.5
→ *Action*: Increase CMA-ES population size, re-optimize

🚩 **Persona drift**: Last turn score < first turn - 1.0
→ *Action*: Investigate conversation history handling

🚩 **Judge disagreement**: Spearman ρ < 0.5
→ *Action*: Use multiple judges, validate with humans

---

## Best Practices

### 1. Prompt Selection

✅ **DO**:
- Use diverse prompts covering multiple topics
- Include open-ended and specific questions
- Balance emotional and neutral prompts

❌ **DON'T**:
- Use only optimization training prompts
- Use prompts too similar to each other
- Use prompts requiring factual knowledge

### 2. Judge Selection

✅ **DO**:
- Use at least 2 different judge models
- Include both GPT and Claude judges
- Test with gpt-4o-mini first (fast/cheap)

❌ **DON'T**:
- Rely on single judge model
- Use models smaller than 7B for judging
- Skip multi-judge validation

### 3. Human Evaluation

✅ **DO**:
- Recruit at least 3 raters per item
- Provide clear persona descriptions
- Randomize response order
- Calculate inter-rater agreement

❌ **DON'T**:
- Use only 1 rater
- Show persona info with labels
- Reveal which is baseline/steered
- Skip agreement calculation

### 4. Reporting

✅ **DO**:
- Report all evaluation metrics
- Include error bars / confidence intervals
- Show both mean and variance
- Discuss limitations

❌ **DON'T**:
- Cherry-pick best results
- Report only mean without variance
- Hide negative results
- Overstate generalization

---

## Using Results in Papers

### Figures to Include

**Figure 1**: Cross-Layer Performance
- Shows persona information distribution in model
- Demonstrates weight transferability
- **Caption**: "Persona fit scores across layers. Weights optimized on layer 20 transfer moderately to adjacent layers, suggesting persona information is distributed hierarchically."

**Figure 2**: Alpha Sensitivity
- Shows robustness to hyperparameters
- Identifies safe operating range
- **Caption**: "Persona fit as a function of steering strength (α). Performance is stable across α ∈ [1.5, 2.5], indicating robust steering."

**Figure 3**: Multi-Turn Stability
- Shows practical usability
- Demonstrates consistency
- **Caption**: "Persona fit over 5-turn conversations. Minimal drift (Δ=-0.15) indicates stable persona maintenance in context."

**Figure 4**: Multi-Judge Agreement
- Validates evaluation methodology
- Shows reliability
- **Caption**: "Inter-judge agreement (Spearman ρ=0.85) across three LLM judges. High correlation validates automated evaluation reliability."

### Key Statistics to Report

```latex
\begin{table}[h]
\centering
\caption{Persona Steering Evaluation Results}
\begin{tabular}{lcc}
\toprule
Metric & episode-184019\_A & Mean (±SD) \\
\midrule
Persona Fit (Train) & 5.00 ± 0.00 & -- \\
Persona Fit (Test)  & 4.67 ± 0.29 & -- \\
Generalization Gap  & 0.33 & < 0.5 \\
Alpha Robustness    & 0.35 & < 0.5 \\
Multi-Turn Drift    & -0.15 & < 0.3 \\
Judge Agreement (ρ) & 0.85 & > 0.7 \\
\bottomrule
\end{tabular}
\end{table}
```

### Writing Tips

**Abstract**:
> "We validate persona reproduction through 5 evaluation dimensions: generalization (train/test gap=0.33), transferability (cross-layer), robustness (α-sensitivity), consistency (multi-turn drift=-0.15), and reliability (inter-judge ρ=0.85)."

**Results Section**:
> "Train/test evaluation reveals minimal overfitting (gap=0.33), indicating weights generalize beyond optimization prompts. Cross-layer analysis shows performance peaks at layer 21 (score=4.90), consistent with optimization target (layer 20, score=5.00). Alpha sensitivity testing across [0.5, 3.0] reveals robust performance (range=0.35), with optimal steering at α=2.0. Multi-turn conversations maintain stable persona fit (drift=-0.15 over 5 turns). Multi-judge evaluation confirms high inter-rater reliability (Spearman ρ=0.85)."

**Limitations Section**:
> "Evaluation uses automated LLM judges. While inter-judge agreement is high (ρ=0.85), human validation with N raters achieves κ=X.XX agreement, confirming automated metrics align with human perception."

---

## Troubleshooting

### Error: "Weights file not found"

**Cause**: Optimization hasn't been run for this persona
**Fix**: Run `scripts/run_persona_optimization.py` first

### Error: "Vector file not found"

**Cause**: SVD vectors missing for some traits/layers
**Fix**: Run `scripts/run_build_svd_vectors.py` for all traits

### Low judge agreement (ρ < 0.5)

**Possible causes**:
1. Judges using different criteria
2. Persona definition is ambiguous
3. Responses are too similar

**Fixes**:
1. Use more specific persona descriptions
2. Increase response diversity (higher temperature)
3. Add human validation

### Persona drift in multi-turn

**Possible causes**:
1. Steering not applied to all turns
2. Context length exceeded
3. Steering interference with conversation history

**Fixes**:
1. Verify steering is active each turn
2. Truncate history if needed
3. Adjust α for conversation mode

---

## Appendix

### File Formats

**result.json structure**:
```json
{
  "evaluation_type": "train_test_split",
  "persona_id": "episode-184019_A",
  "timestamp": "2025-12-08T10:30:00",
  "summary": {
    "train_mean": "5.00",
    "test_mean": "4.67",
    "generalization_gap": "0.33"
  },
  "train": {
    "mean_score": 5.0,
    "std_score": 0.0,
    "scores": [5.0, 5.0, ...]
  },
  "test": {
    "mean_score": 4.67,
    "std_score": 0.29,
    "scores": [5.0, 4.5, 4.5]
  }
}
```

### Dependencies

```txt
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
openai>=1.0.0
anthropic>=0.18.0
```

### Citation

If you use this evaluation framework, please cite:

```bibtex
@software{persona_eval_2025,
  title={Persona Steering Evaluation Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/yourrepo/persona2}
}
```

---

**Last Updated**: 2025-12-08
**Maintainer**: Research Team
**License**: MIT
