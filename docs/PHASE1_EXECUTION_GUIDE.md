# Phase 1 Execution Guide

## Status Overview

### ✓ Completed Tasks

1. **Script Implementation**
   - ✓ Multi-seed aggregation script (`scripts/aggregate_multiseed_results.py`)
   - ✓ Cross-layer evaluation script (`scripts/run_cross_layer_evaluation.py`)
   - ✓ Layer heatmap generation (`scripts/generate_layer_heatmap.py`)
   - ✓ Ablation study script (`scripts/run_ablation_study.py`)
   - ✓ Phase 1 figures generation (`scripts/generate_phase1_figures.py`)
   - ✓ Phase 1 report generation (`scripts/generate_phase1_report.py`)
   - ✓ Phase 2 script templates (TruthfulQA, MMLU, Multi-judge)

2. **Background Processes (In Progress)**
   - ⏳ Baseline comparison seed 2 (PID: 3962196)
   - ⏳ Baseline comparison seed 3 (PID: 3962991)

---

## Step-by-Step Execution Plan

### Step 1: Wait for Baseline Comparisons to Complete

**Monitor progress:**
```bash
./scripts/monitor_phase1_progress.sh
```

**Check process status:**
```bash
ps aux | grep run_baseline_comparison.py | grep -v grep
```

**Expected completion time:**
- Seed 2 & 3: ~2-4 hours each (depending on Grid/Random Search)

**Completion indicators:**
- Processes no longer appear in `ps` output
- Files exist:
  - `reports/experiments/results/baseline_comparison_seed2.json`
  - `reports/experiments/results/baseline_comparison_seed3.json`

---

### Step 2: Generate Integrated Table1 (Multi-seed)

Once seeds 1, 2, 3 are all complete:

```bash
python -B scripts/aggregate_multiseed_results.py
```

**Output:**
- `reports/experiments/tables/table1_multiseed_final.md`
- `reports/experiments/tables/table1_multiseed_final.csv`
- `reports/experiments/tables/table1_multiseed_final.tex`

---

### Step 3: Run Cross-Layer Evaluation

Evaluate all methods across layers 20-24:

```bash
# This will take several hours - recommend running in background
nohup python -B scripts/run_cross_layer_evaluation.py \
  --persona-id episode-184019_A \
  --prompts-file persona-opt/episode-184019_A/eval_prompts.json \
  --num-prompts 20 \
  --seed 1 \
  --methods Base Prompt_Persona MeanDiff PCA Random_Search Grid_Search Proposed \
  --layers 20 21 22 23 24 \
  > logs/phase1/cross_layer_eval.log 2>&1 &
```

**Output:**
- `reports/experiments/results/cross_layer_eval_seed1_<timestamp>.json`

**Estimated time:** 3-5 hours (7 methods × 5 layers × 20 prompts)

---

### Step 4: Generate Layer Heatmap

Once cross-layer evaluation is complete:

```bash
python -B scripts/generate_layer_heatmap.py \
  --results-file reports/experiments/results/cross_layer_eval_seed1_<timestamp>.json
```

**Output:**
- `reports/experiments/figures/layer_heatmap.png`
- `reports/experiments/figures/layer_heatmap_data.csv`

---

### Step 5: Run Ablation Study

```bash
# Recommend running in background
nohup python -B scripts/run_ablation_study.py \
  --persona-id episode-184019_A \
  --prompts-file persona-opt/episode-184019_A/eval_prompts.json \
  --num-prompts 10 \
  --seed 1 \
  --layer 22 \
  > logs/phase1/ablation_study.log 2>&1 &
```

**Output:**
- `reports/experiments/results/ablation_study_seed1_<timestamp>.json`
- `reports/experiments/tables/table_ablation.md`

**Estimated time:** 1-2 hours

---

### Step 6: Generate All Phase 1 Figures

Once ablation is complete:

```bash
# Generate ablation bar chart
python -B scripts/generate_phase1_figures.py \
  --ablation-results reports/experiments/results/ablation_study_seed1_<timestamp>.json \
  --generate-ablation

# Generate seed variation plot
python -B scripts/generate_phase1_figures.py \
  --generate-seed-variation
```

**Output:**
- `reports/experiments/figures/ablation_bar_chart.png`
- `reports/experiments/figures/seed_variation_plot.png`

---

### Step 7: Generate Phase 1 Final Report

```bash
python -B scripts/generate_phase1_report.py
```

**Output:**
- `reports/experiments/phase1_final_report.md`

---

## Quick Commands Reference

### Check what's running
```bash
./scripts/monitor_phase1_progress.sh
```

### Check specific log files
```bash
tail -f logs/phase1/baseline_seed2.log
tail -f logs/phase1/baseline_seed3.log
tail -f logs/phase1/cross_layer_eval.log
tail -f logs/phase1/ablation_study.log
```

### List generated results
```bash
ls -lht reports/experiments/results/
ls -lht reports/experiments/tables/
ls -lht reports/experiments/figures/
```

---

## Troubleshooting

### If baseline comparison hangs

1. Check GPU memory:
   ```bash
   nvidia-smi
   ```

2. Kill and restart:
   ```bash
   kill <PID>
   nohup python -B scripts/run_baseline_comparison.py \
     --persona-id episode-184019_A \
     --prompts-file persona-opt/episode-184019_A/eval_prompts.json \
     --num-prompts 20 \
     --seed <SEED> \
     > logs/phase1/baseline_seed<SEED>.log 2>&1 &
   ```

### If cross-layer evaluation fails

- Check that all required vector files exist:
  ```bash
  ls persona-opt/episode-184019_A/*.pt
  ```

- Required files:
  - `meandiff_vectors.pt`
  - `pca_vectors.pt`
  - `random_search_vectors.pt`
  - `grid_search_vectors.pt`
  - `optimized_vectors.pt`

### If figure generation fails

- Install missing dependencies:
  ```bash
  pip install matplotlib seaborn pandas numpy
  ```

---

## Expected Timeline

| Task | Duration | Status |
|------|----------|--------|
| Baseline seed 2 & 3 | 2-4 hours | ⏳ In Progress |
| Multi-seed aggregation | < 1 min | ⏸ Pending |
| Cross-layer evaluation | 3-5 hours | ⏸ Pending |
| Heatmap generation | < 1 min | ⏸ Pending |
| Ablation study | 1-2 hours | ⏸ Pending |
| Figure generation | < 1 min | ⏸ Pending |
| Report generation | < 1 min | ⏸ Pending |
| **Total** | **6-11 hours** | |

---

## Phase 2 Preparation

Phase 2 script templates are ready in:
- `scripts/run_truthfulqa_eval.py`
- `scripts/run_mmlu_eval.py`
- `scripts/run_multi_judge_eval.py`

These require additional setup:
1. Dataset integration (HuggingFace datasets)
2. Additional judge implementations (GPT-4, Claude)
3. Answer extraction logic for MMLU/TruthfulQA

---

## Notes

- All scripts include timestamp in output filenames for versioning
- All long-running tasks should be run with `nohup` and logged
- Monitor GPU memory usage during experiments
- Results are incrementally saved to prevent data loss
- Random seeds are configurable for reproducibility

---

**Last Updated:** 2025-12-09
