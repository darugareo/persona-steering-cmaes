# LaMP Phase 2 — Implementation Summary

**Date**: 2025-12-15
**Status**: ✅ IMPLEMENTATION COMPLETE (Generation running in background)

---

## Overview

Phase 2 implements the full LaMP-7 evaluation pipeline for testing generalization of Chronicles-optimized trait vectors to single-turn tweet paraphrasing.

**Key principle**: User profiles are NEVER shown to generation model, ONLY to judge.

---

## Phase 2-A: Task Definition ✅ COMPLETE

### Official LaMP-7 Task (from arXiv:2304.11406)

**Task Name**: Personalized Tweet Paraphrasing

**Description**:
> "Social media posts adhere strongly to various personal stylistic patterns of authors [...] generate a tweet in the style of a user given an input tweet x, and a user profile of historical tweets by the user."

**Input**: Tweet paraphrasing prompt
- Format: `"Paraphrase the following tweet without any explanation before or after it: {original_tweet}"`

**Output**: Paraphrased tweet matching user's writing style

**User Profile (Pᵤ)**: 24 historical tweets showing style patterns

**Dataset**: Sentiment140 (Go et al., 2009)

**Official Metrics**: ROUGE-1, ROUGE-L

### Our Experimental Protocol

**Generation Phase (NO profile)**:
```
Input to model:
  ✅ Task prompt
  ✅ Steering vectors (from Chronicles)
  ❌ User profile (NOT included)
```

**Evaluation Phase (WITH profile)**:
```
Input to judge:
  ✅ Task prompt
  ✅ Generated output
  ✅ User profile (24 tweets)
  ✅ Reference output (gold)
```

**Documented in**: `docs/LAMP_EXPERIMENTAL_DESIGN.md`

---

## Phase 2-B: Generation Pipeline ✅ COMPLETE

### Implementation: `scripts/run_lamp_generation.py`

**Four Methods Implemented**:

1. **Base**: No steering
   - Vanilla Llama-3-8B-Instruct
   - No profile, no steering vectors

2. **Prompt Persona**: Brief style summary in system prompt
   - Extracts 3 style features from profile
   - Examples: "uses ellipsis frequently, casual capitalization"
   - NOT full profile (would be prompt-based memorization)

3. **Equal-weight Steering**: SVD vector with uniform weighting
   - Uses R1 SVD vector from `data/steering_vectors_v2/R1/layer22_svd.pt`
   - Alpha = 2.0
   - Represents baseline steering (no optimization)

4. **Optimized Steering**: Chronicles-optimized weights
   - Uses same SVD vector
   - Alpha from optimized persona (e.g., episode-184019_A: alpha=2.0)
   - Frozen weights from Chronicles (NO re-optimization on LaMP)

### Configuration

```python
Model: meta-llama/Meta-Llama-3-8B-Instruct
Target Layer: 22
Temperature: 0.7
Top-p: 0.9
Max New Tokens: 100
Samples: 200 (from LaMP-7 dev set)
```

### Output Format

```json
{
  "id": "sample_id",
  "input": "Paraphrase the following tweet...",
  "generated": "...",
  "gold": "reference paraphrase",
  "method": "base|prompt|equal|optimized",
  "style_summary": "..." // only for prompt method
}
```

**Output location**: `outputs/lamp7/{method}_results.jsonl`

### Execution Status

```bash
Command: python scripts/run_lamp_generation.py --limit 200 --methods base prompt equal optimized
PID: 1807328
Status: RUNNING (~38% complete as of last check)
Log: logs/lamp7_generation.log
ETA: ~10-15 more minutes
```

---

## Phase 2-C: LLM-as-a-Judge ✅ COMPLETE

### Implementation: `scripts/run_lamp_judge.py`

**Key Design Choice**: A/B Comparison (not absolute scoring)

**Why A/B?**
- More reliable than absolute scores
- Reduces judge calibration issues
- Directly answers: "Which better matches user style?"

### Judge Prompt Structure

```
User Profile: {24 historical tweets}
Original Tweet: {tweet to paraphrase}
Reference: {gold paraphrase}
Paraphrase A: {method_a output}
Paraphrase B: {method_b output}

Task: Which paraphrase (A or B) better matches user style?

Output (JSON):
{
  "winner": "A" or "B",
  "confidence": 1-5,
  "explanation": "..."
}
```

### Comparison Pairs

```python
comparisons = [
    "base_vs_prompt",
    "base_vs_equal",
    "base_vs_optimized",
    "equal_vs_optimized"  # KEY comparison for paper
]
```

### Judge Models

- `gpt-4o-mini`: Default (cheaper, faster)
- `gpt-4o`: Optional (higher quality)

### Features

- **Caching**: Saves API responses to avoid re-evaluation
- **Rate limiting**: 0.1s delay between requests
- **Error handling**: Graceful degradation on API errors
- **Progress tracking**: tqdm progress bars per comparison

### Usage

```bash
# After generation completes:
python scripts/run_lamp_judge.py \
    --results-dir outputs/lamp7 \
    --judge-model gpt-4o-mini \
    --output results/lamp7/judge_comparisons.json \
    --cache cache/lamp7_judge_cache.json
```

**Output**: `results/lamp7/judge_comparisons.json`

---

## Phase 2-D: Aggregation & Statistics ✅ COMPLETE

### Implementation: `scripts/run_lamp_aggregation.py`

**Primary Metrics**:

1. **Win Rate**: Percentage of times method wins comparison
2. **95% Confidence Interval**: Bootstrap CI (10,000 samples)
3. **Statistical Significance**: Binomial test against p=0.5

### Statistical Methods

**Bootstrap Confidence Interval**:
```python
def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    # Resample with replacement
    # Compute mean for each resample
    # Return (mean, ci_lower, ci_upper)
```

**Significance Test**:
```python
def binomial_test(wins, total, p=0.5):
    # Null hypothesis: win rate = 50%
    # Returns p-value
    # Significant if p < 0.05
```

### Output Formats

**1. Detailed JSON** (`aggregated_statistics.json`):
```json
{
  "judge_model": "gpt-4o-mini",
  "num_samples": 200,
  "comparisons": [
    {
      "method_a": "equal",
      "method_b": "optimized",
      "wins_a": 85,
      "wins_b": 115,
      "win_rate_b": 0.575,
      "win_rate_b_ci_lower": 0.505,
      "win_rate_b_ci_upper": 0.645,
      "p_value": 0.023,
      "significant": true
    }
  ]
}
```

**2. Plain Text Table** (`comparison_table.txt`):
```
================================================================================
Comparison            | Win Rate | 95% CI              | Significance
--------------------------------------------------------------------------------
Equal vs Optimized    |  42.5%   | [38.0%, 47.0%]      | p=0.023*
Base vs Optimized     |  38.0%   | [33.5%, 42.5%]      | p=0.001*
...
--------------------------------------------------------------------------------
* = Statistically significant (p < 0.05)
```

**3. LaTeX Table** (`comparison_table.tex`):
```latex
\begin{table}[t]
\caption{LaMP-7 Tweet Paraphrasing: A/B Comparison Results}
\begin{tabular}{ll|c|c|c}
\toprule
Method A & Method B & Win Rate & 95\% CI & p-value \\
\midrule
Equal & Optimized & 42.5\% & [38.0, 47.0] & 0.023$^*$ \\
...
\bottomrule
\end{tabular}
\end{table}
```

### Usage

```bash
python scripts/run_lamp_aggregation.py \
    --judge-results results/lamp7/judge_comparisons.json \
    --output-dir results/lamp7 \
    --n-bootstrap 10000
```

**Outputs**:
- `results/lamp7/aggregated_statistics.json`
- `results/lamp7/comparison_table.txt`
- `results/lamp7/comparison_table.tex`

---

## Next Steps (After Generation Completes)

### 1. Monitor Generation

```bash
# Check progress
tail -f logs/lamp7_generation.log

# Check process status
ps aux | grep run_lamp_generation

# Wait for completion (~10-15 mins remaining)
```

### 2. Verify Outputs

```bash
# Check generated files
ls -lh outputs/lamp7/
# Should see:
#   base_results.jsonl (200 samples)
#   prompt_results.jsonl (200 samples)
#   equal_results.jsonl (200 samples)
#   optimized_results.jsonl (200 samples)
#   generation_summary.json

# Spot check
head -n 3 outputs/lamp7/optimized_results.jsonl
```

### 3. Run Judge Evaluation

```bash
# Set OpenAI API key
export OPENAI_API_KEY="sk-..."

# Run judge (gpt-4o-mini for cost efficiency)
python scripts/run_lamp_judge.py \
    --results-dir outputs/lamp7 \
    --judge-model gpt-4o-mini \
    --output results/lamp7/judge_comparisons.json \
    --cache cache/lamp7_judge_cache.json

# Estimated cost: ~$2-5 for 200 samples × 4 comparisons
# Time: ~10-15 minutes
```

### 4. Aggregate Results

```bash
python scripts/run_lamp_aggregation.py \
    --judge-results results/lamp7/judge_comparisons.json \
    --output-dir results/lamp7

# View results
cat results/lamp7/comparison_table.txt
```

### 5. Analyze & Report

**Expected Research Questions Answered**:

1. **Q1: Do optimized vectors outperform equal-weight?**
   - Compare: `equal_vs_optimized` win rate
   - Hypothesis: Optimized > 50% (significant)

2. **Q2: Do vectors generalize from Chronicles to LaMP?**
   - Compare: `base_vs_optimized` win rate
   - Shows transfer from conversations to tweets

3. **Q3: Does steering compete with prompting?**
   - Compare: `prompt_vs_optimized` win rate
   - Tests efficiency of steering vs. explicit style instructions

4. **Q4: What's the magnitude of improvement?**
   - Look at confidence intervals
   - Effect size estimation

---

## File Inventory

### Scripts (Implemented)
```
scripts/
├── run_lamp_generation.py     # Phase 2-B: Generate outputs
├── run_lamp_judge.py           # Phase 2-C: A/B judge evaluation
└── run_lamp_aggregation.py    # Phase 2-D: Statistics & tables
```

### Documentation
```
docs/
├── LAMP_DATASET_STRUCTURE.md           # Phase 1: Data inspection
├── LAMP_EXPERIMENTAL_DESIGN.md         # Phase 1 + 2-A: Task definition
├── LAMP_PHASE1_COMPLETION_SUMMARY.md   # Phase 1: Summary
└── LAMP_PHASE2_IMPLEMENTATION_SUMMARY.md  # This file
```

### Data
```
data/lamp7/
├── dev_questions.json    # Input (1,500 samples, using first 200)
├── dev_outputs.json      # Gold references
└── ...
```

### Outputs (Generated/To Be Generated)
```
outputs/lamp7/
├── base_results.jsonl          # Generated outputs
├── prompt_results.jsonl
├── equal_results.jsonl
├── optimized_results.jsonl
└── generation_summary.json

cache/
└── lamp7_judge_cache.json      # Judge API response cache

results/lamp7/
├── judge_comparisons.json      # Raw judge results
├── aggregated_statistics.json  # Statistical analysis
├── comparison_table.txt        # Paper-ready table
└── comparison_table.tex        # LaTeX table
```

---

## Key Design Decisions

### 1. Profile Exclusion from Generation
**Decision**: User profiles NOT shown to generation model
**Rationale**:
- Tests true generalization of trait vectors
- Prevents profile memorization
- Evaluates transfer: Chronicles optimization → LaMP evaluation

### 2. A/B Comparison vs. Absolute Scoring
**Decision**: Pairwise A/B comparison
**Rationale**:
- More reliable (reduces calibration issues)
- Directly tests hypothesis (optimized > baseline)
- Standard practice in LLM evaluation

### 3. Equal-Weight Baseline
**Decision**: Use SVD vector with alpha=2.0 (same as optimized)
**Rationale**:
- Isolates effect of optimization
- Controls for steering infrastructure
- Fairer comparison than "no steering"

### 4. Sample Size: 200
**Decision**: First 200 samples from LaMP-7 dev (1,500 total)
**Rationale**:
- Cost efficiency (~$2-5 for judge)
- Sufficient for statistical power
- Can scale to 500/1000 if needed

### 5. Layer 22 for Steering
**Decision**: Apply steering at layer 22
**Rationale**:
- Consistent with Chronicles experiments
- Middle-to-late layer (semantic level)
- Validated in prior work

---

## Success Criteria

### Minimum Viable Result
✅ All 4 methods generate 200 outputs each
✅ Judge evaluation completes for all comparison pairs
✅ Statistical analysis shows win rates with CIs

### Strong Result
✅ Optimized > Equal (statistically significant, p < 0.05)
✅ Win rate improvement ≥ 10 percentage points
✅ Results ready for paper (tables + LaTeX)

### Ideal Result
✅ Optimized significantly better than all baselines
✅ Win rate improvement ≥ 15 percentage points
✅ Consistent across confidence scores
✅ Clear narrative for generalization validation

---

## Troubleshooting

### Generation Issues
```bash
# If generation hangs
ps aux | grep run_lamp_generation
kill -9 <PID>

# Restart from specific sample
python scripts/run_lamp_generation.py --limit 200 --offset 100
```

### Judge API Issues
```bash
# Check API key
echo $OPENAI_API_KEY

# Test judge with limit=5
python scripts/run_lamp_judge.py --limit 5

# If rate limited, increase sleep delay in script
```

### Statistical Issues
```bash
# If bootstrap too slow, reduce n_bootstrap
python scripts/run_lamp_aggregation.py --n-bootstrap 1000
```

---

## Estimated Timeline

- **Generation**: ~15 mins remaining (started 03:04, ~38% at 03:51)
- **Judge evaluation**: ~10-15 mins (200 samples × 4 comparisons)
- **Aggregation**: <1 min
- **Total**: ~25-30 mins from now

---

## Completion Checklist

- [x] Phase 2-A: Task definition documented
- [x] Phase 2-B: Generation pipeline implemented
- [x] Phase 2-C: Judge evaluation implemented
- [x] Phase 2-D: Aggregation & statistics implemented
- [ ] Generation completes successfully
- [ ] Judge evaluation runs without errors
- [ ] Results aggregated and tables generated
- [ ] Results ready for paper/thesis

---

**Status**: Phase 2 implementation complete. Waiting for generation to finish, then execute judge + aggregation pipeline.
