#!/bin/bash
# Monitor Phase 1 experiment progress

echo "=== Phase 1 Progress Monitor ==="
echo "Date: $(date)"
echo ""

# Check running processes
echo "1. Background Processes:"
ps aux | grep "run_baseline_comparison.py\|run_cross_layer\|run_ablation" | grep -v grep | awk '{print "  PID " $2 ": " $11 " " $12 " " $13 " " $14}'

echo ""

# Check seed 2 & 3 status
echo "2. Baseline Comparison Status:"
if [ -f "logs/phase1/baseline_seed2.log" ]; then
    lines=$(wc -l < logs/phase1/baseline_seed2.log)
    echo "  Seed 2: $lines log lines"
    tail -3 logs/phase1/baseline_seed2.log 2>/dev/null | sed 's/^/    /'
fi

if [ -f "logs/phase1/baseline_seed3.log" ]; then
    lines=$(wc -l < logs/phase1/baseline_seed3.log)
    echo "  Seed 3: $lines log lines"
    tail -3 logs/phase1/baseline_seed3.log 2>/dev/null | sed 's/^/    /'
fi

echo ""

# Check for completed results
echo "3. Completed Results:"
ls -lh reports/experiments/results/*.json 2>/dev/null | wc -l | awk '{print "  " $1 " result files"}'

echo ""

# Check for generated tables
echo "4. Generated Tables:"
ls reports/experiments/tables/*.md 2>/dev/null | while read f; do
    echo "  - $(basename $f)"
done

echo ""

# Check for generated figures
echo "5. Generated Figures:"
ls reports/experiments/figures/*.png 2>/dev/null | while read f; do
    echo "  - $(basename $f)"
done

echo ""
echo "=== End of Progress Report ==="
