#!/bin/bash
# Automatic continuation of Phase 1 after seeds 2 & 3 complete
# This script waits for seed2 and seed3 to finish, then runs the full pipeline

set -e

BASE_DIR="/data01/nakata/master_thesis/persona2"
cd "$BASE_DIR"

echo "========================================"
echo "Phase 1 Auto-Continuation Script"
echo "========================================"
echo ""
echo "Waiting for baseline comparisons to complete..."
echo "Start time: $(date)"
echo ""

# PIDs to monitor
SEED2_PID=3962196
SEED3_PID=3962991

# Function to check if process is running
is_running() {
    ps -p $1 > /dev/null 2>&1
    return $?
}

# Wait for seed 2
echo "[1/2] Waiting for seed 2 (PID: $SEED2_PID)..."
while is_running $SEED2_PID; do
    sleep 60  # Check every minute
    echo "  $(date +%H:%M:%S) - Seed 2 still running..."
done
echo "✓ Seed 2 completed at $(date)"
echo ""

# Wait for seed 3
echo "[2/2] Waiting for seed 3 (PID: $SEED3_PID)..."
while is_running $SEED3_PID; do
    sleep 60  # Check every minute
    echo "  $(date +%H:%M:%S) - Seed 3 still running..."
done
echo "✓ Seed 3 completed at $(date)"
echo ""

# Verify results exist
echo "Verifying results..."
if [ ! -f "reports/experiments/results/baseline_comparison_seed2.json" ]; then
    echo "ERROR: Seed 2 results not found!"
    exit 1
fi

if [ ! -f "reports/experiments/results/baseline_comparison_seed3.json" ]; then
    echo "ERROR: Seed 3 results not found!"
    exit 1
fi

echo "✓ Both seed results verified"
echo ""

# Wait a bit to ensure files are fully written
sleep 10

echo "========================================"
echo "Starting Phase 1 Automated Pipeline"
echo "========================================"
echo ""

# Run the automated pipeline
./scripts/run_phase1_pipeline.sh

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Phase 1 Complete!"
    echo "========================================"
    echo ""
    echo "Completion time: $(date)"
    echo ""
    echo "Generated files:"
    echo ""
    echo "Tables:"
    ls -lh reports/experiments/tables/*.md 2>/dev/null | tail -5
    echo ""
    echo "Figures:"
    ls -lh reports/experiments/figures/*.png 2>/dev/null
    echo ""
    echo "Final report:"
    echo "  reports/experiments/phase1_final_report.md"
    echo ""
    echo "View report:"
    echo "  cat reports/experiments/phase1_final_report.md"
    echo ""
else
    echo ""
    echo "========================================"
    echo "ERROR: Pipeline failed with exit code $EXIT_CODE"
    echo "========================================"
    echo ""
    echo "Check logs in logs/phase1/"
    exit $EXIT_CODE
fi
