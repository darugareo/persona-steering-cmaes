# CMA-ES Persona Optimization Guide

**Complete guide for optimizing trait weights for persona reproduction using CMA-ES.**

---

## Overview

This system uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to optimize the weights of 5 trait vectors (R1-R5) for each persona, maximizing persona fit as evaluated by the persona-aware judge.

**Objective Function:**
```
steering_vec = w1×R1 + w2×R2 + w3×R3 + w4×R4 + w5×R5
score = mean(persona_fit_scores) across evaluation prompts
CMA-ES minimizes: -score
```

**Trait Definitions:**
- **R1**: Self-Other Focus
- **R2**: Formality
- **R3**: Relationship Distance/Intimacy
- **R4**: Narrative Mode (memory-sharing vs problem-solving)
- **R5**: Emotional Tone

---

## Prerequisites

✅ SVD trait vectors generated (Step 3 completed)
✅ Persona-aware judges created for target personas
✅ OpenAI API key configured in `.env`

**Required Python packages:**
```bash
pip install cma
```

---

## Quick Start

### Test Run (3 iterations, 3 prompts)

```bash
python scripts/test_cmaes_optimization.py
```

This runs a minimal test to verify the pipeline works.

### Full Optimization (Single Persona)

```bash
python scripts/run_persona_optimization.py \
  --persona-id episode-184019_A \
  --layer 20 \
  --max-iterations 30 \
  --num-prompts 10 \
  --alpha 2.0
```

### Batch Optimization (All 3 Personas)

```bash
for PERSONA in episode-184019_A episode-239427_A episode-118328_B; do
  python scripts/run_persona_optimization.py \
    --persona-id $PERSONA \
    --layer 20 \
    --max-iterations 30 \
    --num-prompts 10 \
    --save-dir optimization_results/
done
```

---

## Command-Line Arguments

### `run_persona_optimization.py`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--persona-id` | str | *required* | Persona identifier |
| `--layer` | int | 20 | Layer to apply steering |
| `--alpha` | float | 2.0 | Steering strength |
| `--max-iterations` | int | 30 | Maximum CMA-ES iterations |
| `--population-size` | int | auto | CMA-ES population size |
| `--sigma0` | float | 1.0 | Initial standard deviation |
| `--num-prompts` | int | 15 | Number of evaluation prompts |
| `--save-dir` | str | optimization_results | Output directory |
| `--model` | str | Meta-Llama-3-8B-Instruct | Model name |
| `--trait-vector-dir` | str | data/steering_vectors_v2 | Trait vector directory |
| `--persona-dir` | str | personas | Persona profile directory |

---

## How It Works

### 1. Initialization

```python
optimizer = CMAESPersonaOptimizer(
    persona_id="episode-184019_A",
    layer=20,
    alpha=2.0,
    eval_prompts=[...]  # 10-15 prompts
)
```

Loads:
- 5 SVD trait vectors for the target layer
- Persona profile and judge prompt
- Evaluation prompts
- Llama-3-8B-Instruct model

### 2. Objective Function

For each weight vector `[w1, w2, w3, w4, w5]`:

1. **Build steering vector**: `vec = Σ(wi × trait_vectors[i])`
2. **Generate baseline responses**: Run model without steering
3. **Generate steered responses**: Apply steering with `alpha × vec`
4. **Evaluate with persona-aware judge**: Compare baseline vs steered
5. **Return**: `-mean(persona_fit_scores)` (negative for minimization)

### 3. CMA-ES Optimization

```python
results = optimizer.optimize(
    max_iterations=30,
    sigma0=1.0,
    save_dir="optimization_results"
)
```

CMA-ES explores weight space to maximize persona fit.

**Each iteration:**
- Generates population of weight vectors
- Evaluates each via objective function
- Updates distribution based on fitness
- Converges to optimal weights

### 4. Output

**Generated files:**
```
optimization_results/
├── episode-184019_A_optimization.json    # Full results + history
└── episode-184019_A_best_weights.json    # Best weights only
```

**optimization.json structure:**
```json
{
  "persona_id": "episode-184019_A",
  "layer": 20,
  "alpha": 2.0,
  "best_weights": {
    "R1": 0.234,
    "R2": -0.891,
    "R3": 1.456,
    "R4": 0.678,
    "R5": -0.123
  },
  "best_score": 4.25,
  "num_iterations": 28,
  "num_evaluations": 672,
  "optimization_history": {
    "iterations": [1, 2, 3, ...],
    "best_scores": [3.2, 3.5, 3.8, ...],
    "best_weights": [...]
  }
}
```

---

## Expected Results

### Convergence

Typical optimization should show:
- **Initial score**: ~2.5-3.0 (neutral/slightly positive)
- **Final score**: 3.5-4.5 (good persona fit)
- **Convergence**: 20-40 iterations

### Interpretation

**Persona fit score** (1-5 scale):
- **1-2**: Poor fit (worse than baseline)
- **2-3**: Slight improvement
- **3-4**: Good fit (noticeable improvement)
- **4-5**: Excellent fit (strong persona reproduction)

**Weight magnitudes**:
- Large positive: Strong trait emphasis
- Near zero: Trait not important for this persona
- Large negative: Inverse trait direction

---

## Example: Episode-184019_A

**Persona characteristics** (from profile):
- Informal, high self-focus
- Anecdote/memory sharing style
- High humor, low empathy
- Husband/Wife context

**Expected optimal weights** (hypothesis):
- **R1** (Self-Other): Positive (self-focused)
- **R2** (Formality): Negative (informal)
- **R3** (Relationship Distance): High positive (intimate)
- **R4** (Narrative Mode): Positive (memory-sharing)
- **R5** (Emotional Tone): Moderate (balanced)

---

## Computational Cost

### Per Evaluation

- Model loading: ~3s (first time only)
- Baseline generation: ~2s per prompt
- Steered generation: ~2s per prompt
- Judge evaluation: ~1s per prompt (OpenAI API)

### Total Time Estimate

**Single persona optimization:**
- 30 iterations
- 10 population size (auto)
- 10 prompts per evaluation
- **~300 evaluations × 30s = 2.5 hours**

**Cost estimate (OpenAI API):**
- ~300 evaluations × $0.0001 per eval = **~$0.03**

---

## Advanced Usage

### Custom Evaluation Prompts

```python
custom_prompts = [
    "Describe a memorable moment from your day.",
    "How do you typically respond to unexpected news?",
    "What's your approach to helping a friend in need?"
]

optimizer = CMAESPersonaOptimizer(
    persona_id="episode-184019_A",
    eval_prompts=custom_prompts
)
```

### Layer Sweep

Optimize across multiple layers to find best layer per persona:

```bash
for LAYER in 20 21 22 23 24; do
  python scripts/run_persona_optimization.py \
    --persona-id episode-184019_A \
    --layer $LAYER \
    --max-iterations 20 \
    --save-dir optimization_results/layer_sweep/
done
```

### Custom Initial Weights

Start from hypothesis weights instead of zeros:

```python
import numpy as np

# Hypothesis: high intimacy + informal + self-focused
initial = np.array([
    1.0,   # R1: self-focused
    -1.0,  # R2: informal
    2.0,   # R3: intimate
    0.5,   # R4: memory-sharing
    0.0    # R5: neutral
])

results = optimizer.optimize(
    initial_weights=initial,
    sigma0=0.5  # Smaller sigma when starting near optimum
)
```

---

## Troubleshooting

### Issue: API Rate Limit

**Error**: `RateLimitError: Too many requests`

**Solution**: Reduce population size or add retry delays:
```python
--population-size 6
```

### Issue: Poor Convergence

**Symptoms**: Score doesn't improve after 20+ iterations

**Solutions**:
1. Increase `sigma0`: `--sigma0 2.0`
2. Check persona judge is working: Run test evaluation manually
3. Try different layer: `--layer 21`
4. Increase prompts: `--num-prompts 15`

### Issue: Out of Memory

**Error**: `CUDA out of memory`

**Solution**: Clear GPU cache between evaluations (already implemented in code)

---

## Next Steps

After optimization completes:

1. **Validate results**: Test optimized weights on held-out prompts
2. **Compare personas**: Analyze weight differences across personas
3. **Layer analysis**: Compare results across layers 20-24
4. **Deploy weights**: Use optimized weights for production steering

---

## Files Reference

**Implementation:**
- `/data01/nakata/master_thesis/persona2/persona_opt/cmaes_persona_optimizer.py`
- `/data01/nakata/master_thesis/persona2/scripts/run_persona_optimization.py`
- `/data01/nakata/master_thesis/persona2/scripts/test_cmaes_optimization.py`

**Inputs:**
- Trait vectors: `data/steering_vectors_v2/{R1-R5}/layer{20-24}_svd.pt`
- Persona judges: `personas/{persona_id}/final_judge_prompt.txt`
- Eval prompts: `data/eval_prompts/persona_eval_prompts_v1.json`

**Outputs:**
- Optimization results: `optimization_results/{persona_id}_optimization.json`
- Best weights: `optimization_results/{persona_id}_best_weights.json`

---

## Summary

**Status**: ✅ **CMA-ES Optimizer Fully Implemented**

**Capabilities:**
1. ✅ Multi-trait weight optimization
2. ✅ Persona-aware evaluation
3. ✅ CMA-ES algorithm integration
4. ✅ Automatic result saving
5. ✅ Optimization history tracking

**Ready for**: Production optimization runs on all 3 personas

---

**Generated**: 2025-12-08
**System Version**: 2.0
