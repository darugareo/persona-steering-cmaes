#!/bin/bash

echo "==========================================="
echo " Seed Run Monitor"
echo "==========================================="
date
echo ""

echo "[1] Running processes:"
ps aux | grep run_baseline_comparison_fast.py | grep -v grep | awk '{print " PID:", $2, "CPU:", $3"%", "MEM:", $4"%", $11, $12, $13}'

echo ""
echo "[2] Log file sizes:"
for SEED in 1 2 3; do
  LOG="logs/phase1/baseline_seed${SEED}.log"
  if [ -f "$LOG" ]; then
    echo " Seed ${SEED}: $(wc -l < $LOG) lines"
  else
    echo " Seed ${SEED}: (no log file yet)"
  fi
done

echo ""
echo "[3] Latest log output:"
for SEED in 1 2 3; do
  LOG="logs/phase1/baseline_seed${SEED}.log"
  if [ -f "$LOG" ]; then
    echo "----- Seed ${SEED} (last 10 lines) -----"
    tail -10 "$LOG"
  fi
done

echo ""
echo "==========================================="
echo " To monitor continuously:"
echo "   watch -n 20 ./scripts/monitor_seed_runs.sh"
echo "==========================================="
