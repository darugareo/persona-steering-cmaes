# Phase 1 Quick Reference

## 🚦 Current Status

- ✅ All scripts implemented
- 🏃 Seed 2 & 3 running (PIDs: 3962196, 3962991)
- ⏸️ Waiting for completion to run automated pipeline

---

## ⚡ Quick Commands

### Check Progress
```bash
./scripts/monitor_phase1_progress.sh
```

### Check Running Processes
```bash
ps aux | grep run_baseline_comparison | grep -v grep
```

### View Logs (Real-time)
```bash
tail -f logs/phase1/baseline_seed2.log
tail -f logs/phase1/baseline_seed3.log
```

### Check if Seeds Complete
```bash
ls -lh reports/experiments/results/baseline_comparison_seed*.json
```

---

## 🎯 When Seeds Complete

### Option 1: Automated (Recommended)
```bash
./scripts/run_phase1_pipeline.sh
```
This runs everything automatically (6-11 hours)

### Option 2: Step-by-Step

1. **Aggregate multi-seed**
   ```bash
   python -B scripts/aggregate_multiseed_results.py
   ```

2. **Cross-layer evaluation** (3-5 hours)
   ```bash
   nohup python -B scripts/run_cross_layer_evaluation.py \
     --persona-id episode-184019_A \
     --prompts-file persona-opt/episode-184019_A/eval_prompts.json \
     --num-prompts 20 --seed 1 \
     --methods Base Prompt_Persona MeanDiff PCA Random_Search Grid_Search Proposed \
     --layers 20 21 22 23 24 \
     > logs/phase1/cross_layer_eval.log 2>&1 &
   ```

3. **Generate heatmap**
   ```bash
   python -B scripts/generate_layer_heatmap.py \
     --results-file reports/experiments/results/cross_layer_eval_seed1_*.json
   ```

4. **Ablation study** (1-2 hours)
   ```bash
   nohup python -B scripts/run_ablation_study.py \
     --persona-id episode-184019_A \
     --prompts-file persona-opt/episode-184019_A/eval_prompts.json \
     --num-prompts 10 --seed 1 --layer 22 \
     > logs/phase1/ablation_study.log 2>&1 &
   ```

5. **Generate figures**
   ```bash
   python -B scripts/generate_phase1_figures.py \
     --ablation-results reports/experiments/results/ablation_study_*.json \
     --generate-ablation

   python -B scripts/generate_phase1_figures.py \
     --generate-seed-variation
   ```

6. **Generate report**
   ```bash
   python -B scripts/generate_phase1_report.py
   ```

---

## 📊 Expected Outputs

### After All Steps Complete

```
reports/experiments/
├── results/
│   ├── baseline_comparison_seed1.json ✅
│   ├── baseline_comparison_seed2.json 🏃
│   ├── baseline_comparison_seed3.json 🏃
│   ├── cross_layer_eval_seed1_*.json ⏸️
│   └── ablation_study_seed1_*.json    ⏸️
├── tables/
│   ├── table1_multiseed_final.md      ⏸️
│   └── table_ablation.md              ⏸️
├── figures/
│   ├── layer_heatmap.png              ⏸️
│   ├── ablation_bar_chart.png         ⏸️
│   └── seed_variation_plot.png        ⏸️
└── phase1_final_report.md             ⏸️
```

---

## 📖 Documentation

- **Full guide:** `docs/PHASE1_EXECUTION_GUIDE.md`
- **Implementation details:** `PHASE1_IMPLEMENTATION_SUMMARY.md`
- **Current status:** `PHASE1_STATUS.md`

---

## ⏱️ Timeline

| Task | Duration |
|------|----------|
| Seeds 2 & 3 (running) | 2-4 hours |
| Aggregation | < 1 min |
| Cross-layer | 3-5 hours |
| Ablation | 1-2 hours |
| Figures | < 1 min |
| Report | < 1 min |
| **Total** | **6-11 hours** |

---

## 🔧 Troubleshooting

### Process hung?
```bash
# Check GPU
nvidia-smi

# Kill and restart
kill <PID>
nohup python -B scripts/run_baseline_comparison.py ... &
```

### Missing files?
```bash
# Check required vectors
ls persona-opt/episode-184019_A/*.pt
```

### Python errors?
```bash
# Check dependencies
pip install matplotlib seaborn pandas numpy torch transformers
```

---

## ✅ Success Criteria

Phase 1 complete when:
- [ ] Seeds 1, 2, 3 all have results
- [ ] Table1 with mean±std generated
- [ ] Layer heatmap shows 7 methods × 5 layers
- [ ] Ablation table shows 8 configurations
- [ ] All 3 figures generated
- [ ] Final report exists

---

## 🚀 Next: Phase 2

After Phase 1 complete:
1. Review `phase1_final_report.md`
2. Archive results with timestamp
3. Implement Phase 2 dataset loaders
4. Set up GPT-4/Claude API access
5. Run TruthfulQA, MMLU, Multi-judge evaluations

---

**Last updated:** 2025-12-09 15:56 JST
