# Lightweight 10-Persona Execution Guide

**Strategy**: Equal-weight for all, optimized for subset
**Estimated time**: 2-3 hours (vs 49+ hours with full optimization)
**Computational cost**: 70% reduction

---

## ✅ Pre-Execution Checklist

### 1. Verify Persona Data
```bash
cd /data01/nakata/master_thesis/persona2

# Check all 7 new personas
for p in episode-5289_A episode-29600_A episode-88279_B episode-132247_A episode-166805_A episode-196697_B episode-225888_A; do
  ls personas/$p/persona_samples.json || echo "Missing: $p"
done
```

Expected: All files exist

### 2. Verify Existing 3 Personas Results
```bash
# Check existing optimized results
for p in episode-184019_A episode-239427_A episode-118328_B; do
  ls results/cross_model/mistral_7b/$p/optimized.jsonl || echo "Missing: $p/optimized"
done
```

Expected: All 3 personas have optimized.jsonl

### 3. Verify Trait Vectors
```bash
ls data/steering_vectors_v2/R{1,2,3,4,5}/layer22_svd.pt
```

Expected: 5 files found

### 4. Set OpenAI API Key
```bash
export OPENAI_API_KEY="your_api_key_here"
```

---

## 🚀 Execution Steps

### Step 1: Generate Responses (7 New Personas, Equal-Weight Only)

**Time**: ~2-3 hours

```bash
cd /data01/nakata/master_thesis/persona2

# Run in tmux for safety
tmux new -s persona_gen

# Execute generation
python experiments/run_7personas_lightweight.py
```

**What it does**:
- Generates responses for 7 new personas
- Methods: base, prompt, equal (NO optimized)
- 28 prompts per method
- Total: 7 × 3 × 28 = 588 generations

**Output**:
```
results/cross_model/mistral_7b/
├── episode-5289_A/
│   ├── base.jsonl
│   ├── prompt.jsonl
│   └── equal.jsonl
├── episode-29600_A/
│   ├── base.jsonl
│   ├── prompt.jsonl
│   └── equal.jsonl
...
```

**Monitor progress**:
```bash
# Detach: Ctrl+B, D
# Reattach: tmux attach -t persona_gen

# Check log
tail -f logs/7personas_lightweight_generation.log
```

---

### Step 2: Judge Evaluation (All 10 Personas)

**Time**: ~4-5 hours
**Depends on**: Step 1 completion + OpenAI API key

```bash
# Verify Step 1 completed
python -c "
from pathlib import Path
count = 0
for p in ['episode-5289_A', 'episode-29600_A', 'episode-88279_B', 'episode-132247_A', 'episode-166805_A', 'episode-196697_B', 'episode-225888_A']:
    for m in ['base', 'prompt', 'equal']:
        if (Path('results/cross_model/mistral_7b') / p / f'{m}.jsonl').exists():
            count += 1
print(f'Found {count}/21 files')
"

# Run judge evaluation
tmux new -s judge_eval
python scripts/run_judge_evaluation_10personas.py
```

**What it does**:
- Compares:
  - All 10 personas: base vs equal, base vs prompt
  - Subset 3 personas: equal vs optimized
- Judge: gpt-4o-mini (75%) + gpt-4o (25% spot check)
- Total comparisons: ~840

**Output**:
```
results/judge_evaluation/10personas_lightweight_results.json
```

**Estimated API cost**:
- ~630 calls to gpt-4o-mini ($0.15 per 1M input tokens)
- ~210 calls to gpt-4o ($2.50 per 1M input tokens)
- Total: $10-15 USD

---

### Step 3: Aggregate Results

**Time**: <5 minutes

```bash
python scripts/aggregate_results_10personas.py
```

**What it does**:
- Computes win rates with 95% CI (bootstrap)
- Statistical tests: McNemar, Sign test
- Generates tables (Markdown + LaTeX)

**Output**:
```
reports/10personas/
├── aggregated_stats.json
└── tables/
    ├── table_main_results.md
    ├── table_main_results.tex
    └── table_per_persona.md
```

---

### Step 4: Update IEEE Access Paper

**Time**: ~30 minutes

#### 4.1 Update Experimental Setup

Add computational cost justification:

**File**: `paper_ieee_access/sections/experimental_setup.tex`

**Location**: After persona selection table

**Content**: See `COMPUTATIONAL_COST_JUSTIFICATION.txt` for exact wording

#### 4.2 Update Results Section

Replace existing results table with new 10-persona table.

**File**: `paper_ieee_access/sections/results.tex`

**Action**: Replace Table 2 with `reports/10personas/tables/table_main_results.tex`

#### 4.3 Update Abstract (if needed)

If win rates changed significantly, update abstract statistics.

**File**: `paper_ieee_access/sections/abstract.tex`

**Current**: "67.5% vs 35.7%, p<0.001"

**Update to**: [new win rate from aggregated results]

#### 4.4 Recompile Paper

```bash
cd paper_ieee_access
pdflatex ieee_access.tex
bibtex ieee_access
pdflatex ieee_access.tex
pdflatex ieee_access.tex
```

---

## 📊 Expected Results

### Main Findings (10 Personas)

**Equal vs Base**:
- Expected win rate: ~60-70%
- Shows trait-based steering works universally

**Prompt vs Base**:
- Expected win rate: ~40-50%
- Shows prompt-based baseline is weak

### Subset Analysis (3 Personas)

**Optimized vs Equal**:
- Expected win rate: ~65-75%
- Shows CMA-ES optimization adds value
- Justifies computational cost for focused study

---

## 🔍 Validation Checklist

### After Step 1 (Generation)
- [ ] All 21 files exist (7 personas × 3 methods)
- [ ] Each file has 28 lines (1 per prompt)
- [ ] No error messages in responses

```bash
# Quick check
for p in episode-5289_A episode-29600_A episode-88279_B episode-132247_A episode-166805_A episode-196697_B episode-225888_A; do
  for m in base prompt equal; do
    count=$(wc -l < results/cross_model/mistral_7b/$p/$m.jsonl 2>/dev/null || echo 0)
    echo "$p/$m: $count lines"
  done
done
```

### After Step 2 (Evaluation)
- [ ] Judge results file exists
- [ ] All comparisons complete
- [ ] Spot check ratio ~25%

```bash
# Check completeness
python -c "
import json
with open('results/judge_evaluation/10personas_lightweight_results.json') as f:
    data = json.load(f)
    for result in data['results']:
        print(f\"{result['method_a']} vs {result['method_b']}: {result['total_comparisons']} comparisons\")
"
```

### After Step 3 (Aggregation)
- [ ] Tables generated (Markdown + LaTeX)
- [ ] Win rates reasonable (40-75%)
- [ ] p-values < 0.05 for key comparisons

```bash
cat reports/10personas/tables/table_main_results.md
```

---

## ⚠️ Troubleshooting

### Issue: Generation fails with CUDA OOM

**Solution**: Reduce batch size or use smaller model

```python
# In run_7personas_lightweight.py, reduce max_new_tokens
GEN_CONFIG = GenerationConfig(
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=100,  # Reduced from 150
    ...
)
```

### Issue: Judge evaluation gets rate limited

**Solution**: Add delay between API calls

```python
# In run_judge_evaluation_10personas.py, add:
import time
time.sleep(1)  # After each API call
```

### Issue: Missing optimized.jsonl for existing 3 personas

**Solution**: Copy from Phase 1-C results or regenerate

```bash
# If Phase 1-C results exist elsewhere
cp /path/to/phase1c/results/cross_model/mistral_7b/*/optimized.jsonl \
   results/cross_model/mistral_7b/
```

---

## 📝 Paper Integration

### Key Phrases for Experimental Setup

> "Due to the high computational cost of evolutionary optimization (approximately 7 GPU-hours per persona), CMA-ES was applied to a representative subset of three personas spanning diverse communication styles. Equal-weight steering was evaluated across all ten personas to assess general applicability."

### Key Phrases for Results

> "Equal-weight steering achieved X% win rate across 10 personas (95% CI [Y, Z], p<0.001), demonstrating robust generalization. Optimization on a representative subset (n=3) improved win rates to W% (p<0.01), validating the value of persona-specific tuning."

### Key Phrases for Discussion

> "The two-tier evaluation strategy (universal equal-weight + subset optimization) balances generalizability testing with computational feasibility. Results show that trait-based steering transfers effectively across personas without optimization, while evolutionary tuning provides measurable improvements for focused applications."

---

## ✅ Success Criteria

### Technical
- [ ] All 588 new responses generated
- [ ] Judge evaluation complete with >800 comparisons
- [ ] Statistical significance (p<0.05) for key comparisons
- [ ] Tables and figures generated

### Paper Quality
- [ ] Computational cost justified clearly
- [ ] Results section updated with 10-persona tables
- [ ] Abstract reflects new statistics
- [ ] Experimental Setup explains two-tier strategy

### Reviewer-Ready
- [ ] No missing data or incomplete experiments
- [ ] Clear rationale for subset optimization
- [ ] Reproducible (seed=42 everywhere)
- [ ] Figures/tables publication-quality

---

## 🎯 Final Checklist Before Submission

- [ ] PDF compiles without errors
- [ ] All tables have correct captions
- [ ] All figures referenced in text
- [ ] Computational cost justification clear
- [ ] No "we/our/I" pronouns (IEEE style)
- [ ] References formatted correctly
- [ ] Supplementary materials prepared (if any)

---

**Total Execution Time**: ~6-8 hours (mostly automated)
**Paper Update Time**: ~30 minutes
**Total Time to Submission-Ready**: <1 day

**Status**: Ready to execute
**Next Command**: `python experiments/run_7personas_lightweight.py`
