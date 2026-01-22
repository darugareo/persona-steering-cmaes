# 7 New Personas - Progress Summary

**Date**: 2025-12-15 19:15 JST
**Status**: Conversation extraction complete, ready for optimization

---

## ✅ Completed Tasks

### 1. Persona Selection & Extraction
- ✅ Identified 7 target personas via stratified numeric sampling
- ✅ Fixed 2 missing personas (episode-31102, episode-229805)
- ✅ Replaced with: episode-29600_A, episode-225888_A
- ✅ Extracted conversation data for all 7 personas
- ✅ Updated paper tables with corrected episode IDs

### 2. Data Validation
All 7 personas successfully extracted with sufficient utterances:

| ID | Persona ID | Relationship | Sessions | Utterances | Avg Length | Status |
|----|------------|--------------|----------|------------|------------|--------|
| P4 | episode-5289_A | Husband and Wife | 5 | 40 | 81.1 | ✅ |
| P5 | episode-29600_A | Parent and Child | 5 | 44 | 90.0 | ✅ |
| P6 | episode-88279_B | Classmates | 5 | 42 | 64.0 | ✅ |
| P7 | episode-132247_A | Neighbors | 5 | 40 | 95.5 | ✅ |
| P8 | episode-166805_A | Neighbors | 5 | 42 | 101.8 | ✅ |
| P9 | episode-196697_B | Classmates | 5 | 43 | 71.8 | ✅ |
| P10 | episode-225888_A | Classmates | 5 | 41 | 102.8 | ✅ |

### 3. Infrastructure
- ✅ Trait vectors (R1-R5) available in `data/steering_vectors_v2/`
- ✅ Optimization script created: `scripts/run_7personas_optimization.py`
- ✅ Extension pipeline script ready: `experiments/run_phase1_extension_7personas.py`

---

## ⏳ Pending Tasks

### 1. CMA-ES Optimization (CRITICAL PATH)
**Time estimate**: ~49 hours (7 personas × 7 hours each)

**What needs to be done**:
- Optimize trait vector weights for each persona using CMA-ES
- Source model: Llama-3-8B-Instruct
- Configuration: layer=20, alpha=2.0, 100 iterations, sigma=0.3

**Command**:
```bash
python scripts/run_7personas_optimization.py
```

**Output**: `persona-opt/{persona_id}/best_weights.json` for each persona

**Recommendation**: Run in tmux/screen session due to long runtime

---

### 2. Cross-Model Generation
**Time estimate**: ~2-3 hours
**Depends on**: CMA-ES optimization completion

**What needs to be done**:
- Generate responses for 7 personas on Mistral-7B
- 4 methods: base, prompt, equal, optimized
- 28 prompts (20 from v1 + 8 from v2)
- Total: 7 × 4 × 28 = 784 generations

**Command**:
```bash
# After optimization completes
python experiments/run_phase1b_generation.py --extended
```

**Output**: `results/cross_model/mistral_7b/{persona_id}/{method}.jsonl`

---

### 3. Judge Evaluation
**Time estimate**: ~4-5 hours
**Depends on**: Cross-model generation completion

**What needs to be done**:
- LLM-as-judge evaluation (gpt-4o-mini as main judge)
- 25% spot check with gpt-4o for reliability
- Pairwise comparisons: optimized vs base/prompt/equal
- Total: 10 personas × 4 methods × 28 prompts = 1,120 comparisons

**Command**:
```bash
python scripts/run_judge_evaluation_10personas.py
```

**Output**: `results/judge_evaluation/10personas_results.json`

---

### 4. Statistical Aggregation
**Time estimate**: <1 hour
**Depends on**: Judge evaluation completion

**What needs to be done**:
- Aggregate results for all 10 personas
- Compute win-rates, 95% CI (bootstrap)
- McNemar test, Sign test for statistical significance
- Generate per-persona and per-category tables

**Command**:
```bash
python scripts/aggregate_results_10personas.py
```

**Output**:
- `reports/10personas_aggregated_results.json`
- `reports/tables/table_10personas_results.tex`
- `reports/tables/table_per_persona.tex`
- `reports/tables/table_per_category.tex`

---

### 5. Paper Updates
**Time estimate**: <1 hour
**Depends on**: Statistical aggregation completion

**What needs to be done**:
- Update Abstract with 10-persona results
- Update Results section with new statistics
- Update all figure captions to say "10 personas"
- Update Experimental Setup to mention 10 personas
- Recompile IEEE Access paper

**Files to update**:
- `paper_ieee_access/sections/abstract.tex` ✅ (already updated with corrected IDs)
- `paper_ieee_access/sections/results.tex`
- `paper_ieee_access/sections/experimental_setup.tex`
- `paper_ieee_access/tables/persona_selection_table.tex` ✅ (already updated)

---

## 📊 Timeline

| Stage | Duration | Start After | Cumulative Time |
|-------|----------|-------------|-----------------|
| ✅ Persona extraction | 30 min | - | 30 min |
| ⏳ CMA-ES optimization | 49 hours | - | ~2 days |
| ⏳ Cross-model generation | 2-3 hours | Optimization | ~2.1 days |
| ⏳ Judge evaluation | 4-5 hours | Generation | ~2.3 days |
| ⏳ Aggregation | 1 hour | Evaluation | ~2.3 days |
| ⏳ Paper updates | 1 hour | Aggregation | ~2.3 days |

**Total estimated time**: ~2.3 days (assuming overnight runs)

---

## 💡 Recommendations

### Option A: Complete 10-Persona Extension (Recommended for Comprehensive Paper)
**Timeline**: ~2.3 days
**Pros**:
- More robust results (3.3× larger sample)
- Demonstrates method scalability
- Stronger statistical power
- Addresses reviewer concerns about limited personas

**Cons**:
- Requires 2+ days of computation
- Risk of failures during long runs

**Action**:
```bash
# Start optimization immediately in tmux
tmux new -s persona_opt
python scripts/run_7personas_optimization.py
# Ctrl+B, D to detach
```

---

### Option B: Submit with 3 Personas, Note Future Extension
**Timeline**: Immediate (paper already complete)
**Pros**:
- Paper ready for submission now
- Phase 1-C results already validated
- Can submit 7-persona extension as follow-up

**Cons**:
- Reviewer may request more personas
- Smaller statistical power (n=3)
- May need major revision

**Action**:
- Add footnote in Experimental Setup mentioning ongoing 10-persona extension
- Submit current 3-persona paper
- Prepare 7-persona extension for revision/follow-up

---

## 🎯 Current Recommendation

**Start CMA-ES optimization immediately** while keeping paper submission options open:

1. **Start overnight run**: Launch CMA-ES for 7 personas (49 hours)
2. **Monitor progress**: Check logs periodically
3. **Parallel track**: Keep 3-persona paper ready for submission
4. **Decision point** (in 2 days):
   - If optimization succeeds → Update to 10-persona paper
   - If failures occur → Submit 3-persona paper with extension note

This maximizes flexibility while making progress toward the stronger 10-persona result.

---

## 📝 Files Generated

### Completed
- ✅ `personas_final_10_corrected.txt` - Final 10 persona IDs (corrected)
- ✅ `PERSONA_UPDATE_NOTE.md` - Documentation of ID changes
- ✅ `paper_ieee_access/tables/persona_selection_table.tex` - Updated table
- ✅ `scripts/extract_7_new_personas.py` - Extraction script (completed)
- ✅ `scripts/run_7personas_optimization.py` - Optimization script (ready)
- ✅ `experiments/run_phase1_extension_7personas.py` - Full pipeline (ready)

### Pending
- ⏳ `persona-opt/{persona_id}/best_weights.json` (7 files)
- ⏳ `results/cross_model/mistral_7b/{persona_id}/*.jsonl` (28 files)
- ⏳ `results/judge_evaluation/10personas_results.json`
- ⏳ `reports/10personas_aggregated_results.json`
- ⏳ `reports/tables/table_10personas_results.tex`

---

**Status**: Ready to start CMA-ES optimization (49-hour critical path)

**Next Command**:
```bash
tmux new -s persona_opt
cd /data01/nakata/master_thesis/persona2
python scripts/run_7personas_optimization.py
```
