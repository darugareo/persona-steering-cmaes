# LaMP-7 Execution Guide

Quick reference for running the full evaluation pipeline after generation completes.

---

## 1. Monitor Generation Progress

```bash
# Check current progress
tail -f logs/lamp7_generation.log

# Check process status
ps aux | grep run_lamp_generation

# When you see "Generation complete!" the process is done
```

---

## 2. Verify Generation Outputs

```bash
# Check all output files created
ls -lh outputs/lamp7/

# Should see:
#   base_results.jsonl (200 samples)
#   prompt_results.jsonl (200 samples)
#   equal_results.jsonl (200 samples)
#   optimized_results.jsonl (200 samples)
#   generation_summary.json

# Count samples in each file
wc -l outputs/lamp7/*.jsonl

# Spot check a few outputs
head -n 3 outputs/lamp7/optimized_results.jsonl | python -m json.tool
```

---

## 3. Run LLM Judge Evaluation

**Prerequisites**:
- OpenAI API key set in environment
- Estimated cost: $2-5 for 200 samples × 4 comparisons with gpt-4o-mini
- Estimated time: 10-15 minutes

```bash
# Set API key (if not already set)
export OPENAI_API_KEY="sk-..."

# Run judge evaluation
python scripts/run_lamp_judge.py \
    --results-dir outputs/lamp7 \
    --judge-model gpt-4o-mini \
    --output results/lamp7/judge_comparisons.json \
    --cache cache/lamp7_judge_cache.json

# The script will:
# - Load all 4 method outputs
# - Run A/B comparisons for each pair
# - Cache results for reproducibility
# - Print progress bars for each comparison
# - Show quick summary at the end
```

**Expected output**:
```
Comparing base vs prompt
base vs prompt: 88/200 (44.0%) A wins, 112/200 (56.0%) B wins

Comparing base vs equal
base vs equal: 95/200 (47.5%) A wins, 105/200 (52.5%) B wins

Comparing base vs optimized
base vs optimized: 72/200 (36.0%) A wins, 128/200 (64.0%) B wins

Comparing equal vs optimized
equal vs optimized: 85/200 (42.5%) A wins, 115/200 (57.5%) B wins
```

---

## 4. Aggregate Results & Generate Tables

```bash
# Run statistical aggregation
python scripts/run_lamp_aggregation.py \
    --judge-results results/lamp7/judge_comparisons.json \
    --output-dir results/lamp7 \
    --n-bootstrap 10000

# The script will:
# - Compute win rates for each comparison
# - Calculate 95% confidence intervals (bootstrap)
# - Run statistical significance tests
# - Generate tables in multiple formats
```

**Expected output files**:
```
results/lamp7/
├── aggregated_statistics.json    # Detailed JSON stats
├── comparison_table.txt          # Plain text table
└── comparison_table.tex          # LaTeX table for paper
```

---

## 5. View Results

```bash
# View plain text table
cat results/lamp7/comparison_table.txt

# View detailed statistics
cat results/lamp7/aggregated_statistics.json | python -m json.tool

# Copy LaTeX table for paper
cat results/lamp7/comparison_table.tex
```

**Expected table format**:
```
================================================================================
LaMP-7 A/B Comparison Results
================================================================================
Judge Model: gpt-4o-mini
Samples: 200
Bootstrap Samples: 10000
================================================================================

Comparison            | Win Rate | 95% CI              | Significance
--------------------------------------------------------------------------------
Base vs Prompt        |  44.0%   | [39.5%, 48.5%]      | p=0.123
Base vs Equal         |  47.5%   | [43.0%, 52.0%]      | p=0.567
Base vs Optimized     |  36.0%   | [31.5%, 40.5%]      | p=0.001*
Equal vs Optimized    |  42.5%   | [38.0%, 47.0%]      | p=0.023*
--------------------------------------------------------------------------------

* = Statistically significant (p < 0.05)
```

---

## 6. Interpret Results

### Key Questions to Answer:

**Q1: Does optimization help?**
- Look at `equal_vs_optimized` comparison
- Optimized should win significantly (> 50%, p < 0.05)
- This shows Chronicles optimization transfers to LaMP

**Q2: How much does it help?**
- Win rate difference (e.g., 57.5% = +15 percentage points over chance)
- Confidence interval width (narrower = more confident)

**Q3: Is steering better than prompting?**
- Look at `prompt_vs_optimized` (if included)
- Shows whether steering is more effective than explicit style instructions

**Q4: Statistical significance**
- p < 0.05 = statistically significant
- Means improvement is unlikely due to chance

---

## 7. Optional: Generate Additional Comparisons

If you want to compare prompt vs optimized:

```bash
# Re-run judge with additional comparison
python scripts/run_lamp_judge.py \
    --results-dir outputs/lamp7 \
    --judge-model gpt-4o-mini \
    --comparisons base_vs_prompt base_vs_equal base_vs_optimized equal_vs_optimized prompt_vs_optimized \
    --output results/lamp7/judge_comparisons_full.json \
    --cache cache/lamp7_judge_cache.json

# Re-run aggregation
python scripts/run_lamp_aggregation.py \
    --judge-results results/lamp7/judge_comparisons_full.json \
    --output-dir results/lamp7_full
```

---

## 8. Full Pipeline (Copy-Paste)

After generation completes, run this complete sequence:

```bash
# 1. Verify outputs
ls -lh outputs/lamp7/
wc -l outputs/lamp7/*.jsonl

# 2. Set API key
export OPENAI_API_KEY="sk-..."

# 3. Run judge
python scripts/run_lamp_judge.py \
    --results-dir outputs/lamp7 \
    --judge-model gpt-4o-mini \
    --output results/lamp7/judge_comparisons.json \
    --cache cache/lamp7_judge_cache.json

# 4. Aggregate results
python scripts/run_lamp_aggregation.py \
    --judge-results results/lamp7/judge_comparisons.json \
    --output-dir results/lamp7

# 5. View results
cat results/lamp7/comparison_table.txt
```

---

## Troubleshooting

### Judge API Errors

```bash
# Test API connection with small sample
python scripts/run_lamp_judge.py --limit 5

# Check API key
echo $OPENAI_API_KEY

# If rate limited, the script will retry automatically
# Cache prevents re-running successful requests
```

### Missing Results

```bash
# Check if generation completed successfully
tail -100 logs/lamp7_generation.log | grep -E "(ERROR|complete)"

# Check file sizes
du -h outputs/lamp7/*

# Re-run generation for specific method if needed
python scripts/run_lamp_generation.py --limit 200 --methods base
```

### Statistical Issues

```bash
# If bootstrap is slow, reduce samples
python scripts/run_lamp_aggregation.py \
    --judge-results results/lamp7/judge_comparisons.json \
    --output-dir results/lamp7 \
    --n-bootstrap 1000  # Default is 10000
```

---

## Next Steps for Paper

1. **Copy results to thesis/paper directory**
   ```bash
   cp results/lamp7/comparison_table.tex ~/thesis/tables/
   cp results/lamp7/aggregated_statistics.json ~/thesis/results/
   ```

2. **Create visualizations** (optional)
   - Bar chart of win rates
   - Confidence interval plots
   - Per-sample confidence distribution

3. **Write results section**
   - Hypothesis: Optimized > Equal-weight
   - Results: Win rate, CI, significance
   - Interpretation: What this means for generalization

4. **Compare with Chronicles results**
   - Does LaMP show similar improvements?
   - Any performance gaps (transfer learning)?

---

**Estimated Total Time**: ~25-30 minutes from generation start to final results
