#!/bin/bash
################################################################################
# Run 26 Personas Optimization in Parallel on 2 GPUs
################################################################################
#
# This script runs CMA-ES optimization for all 26 personas in parallel:
# - GPU 0: Personas 0-12 (13 personas)
# - GPU 1: Personas 13-25 (13 personas)
#
# Usage:
#   bash run_all_26personas_parallel.sh
#
# Monitor progress:
#   tail -f gpu0_log.txt
#   tail -f gpu1_log.txt
#
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "26 PERSONAS PARALLEL OPTIMIZATION"
echo "================================================================================"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  GPU 0: Personas 0-12 (13 personas)"
echo "  GPU 1: Personas 13-25 (13 personas)"
echo "  Max iterations: 10"
echo "  Population size: 8"
echo "  Eval prompts: 10 (for speed)"
echo ""
echo "Estimated time: 4-7 hours per GPU"
echo "================================================================================"
echo ""

# Check if personas_final_26.txt exists
if [ ! -f "personas_final_26.txt" ]; then
    echo "ERROR: personas_final_26.txt not found!"
    exit 1
fi

# Count personas
PERSONA_COUNT=$(wc -l < personas_final_26.txt)
echo "✓ Found $PERSONA_COUNT personas in personas_final_26.txt"

if [ "$PERSONA_COUNT" -ne 26 ]; then
    echo "WARNING: Expected 26 personas, found $PERSONA_COUNT"
fi

# Create log directory
mkdir -p logs
LOG_DIR="logs/optimization_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo ""
echo "Log directory: $LOG_DIR"
echo ""

# GPU 0 command
GPU0_CMD="CUDA_VISIBLE_DEVICES=0 python3 scripts/run_26personas_batch_gpu.py \
    --gpu_id 0 \
    --start 0 \
    --end 13 \
    --layer 20 \
    --alpha 2.0 \
    --max-iterations 10 \
    --population-size 8 \
    --num-prompts 10 \
    --save-dir optimization_results_26personas"

# GPU 1 command
GPU1_CMD="CUDA_VISIBLE_DEVICES=1 python3 scripts/run_26personas_batch_gpu.py \
    --gpu_id 1 \
    --start 13 \
    --end 26 \
    --layer 20 \
    --alpha 2.0 \
    --max-iterations 10 \
    --population-size 8 \
    --num-prompts 10 \
    --save-dir optimization_results_26personas"

# Start GPU 0 in background
echo "🚀 Starting GPU 0 (personas 0-12)..."
echo "   Log: $LOG_DIR/gpu0.log"
nohup bash -c "$GPU0_CMD" > "$LOG_DIR/gpu0.log" 2>&1 &
GPU0_PID=$!
echo "   PID: $GPU0_PID"

# Wait a bit to avoid simultaneous model loading
sleep 5

# Start GPU 1 in background
echo "🚀 Starting GPU 1 (personas 13-25)..."
echo "   Log: $LOG_DIR/gpu1.log"
nohup bash -c "$GPU1_CMD" > "$LOG_DIR/gpu1.log" 2>&1 &
GPU1_PID=$!
echo "   PID: $GPU1_PID"

# Save PIDs
echo "$GPU0_PID" > "$LOG_DIR/gpu0.pid"
echo "$GPU1_PID" > "$LOG_DIR/gpu1.pid"

echo ""
echo "================================================================================"
echo "BOTH GPUS STARTED"
echo "================================================================================"
echo ""
echo "GPU 0 PID: $GPU0_PID (personas 0-12)"
echo "GPU 1 PID: $GPU1_PID (personas 13-25)"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_DIR/gpu0.log"
echo "  tail -f $LOG_DIR/gpu1.log"
echo ""
echo "Check if running:"
echo "  ps aux | grep $GPU0_PID"
echo "  ps aux | grep $GPU1_PID"
echo ""
echo "Stop all:"
echo "  kill $GPU0_PID $GPU1_PID"
echo ""
echo "Results will be saved to:"
echo "  optimization_results_26personas/gpu0/"
echo "  optimization_results_26personas/gpu1/"
echo ""
echo "================================================================================"
echo "Waiting for completion... (this may take 4-7 hours)"
echo "================================================================================"

# Wait for both processes
wait $GPU0_PID
GPU0_EXIT=$?

wait $GPU1_PID
GPU1_EXIT=$?

echo ""
echo "================================================================================"
echo "OPTIMIZATION COMPLETE"
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "GPU 0 exit code: $GPU0_EXIT"
echo "GPU 1 exit code: $GPU1_EXIT"
echo ""

if [ $GPU0_EXIT -eq 0 ] && [ $GPU1_EXIT -eq 0 ]; then
    echo "✅ Both GPUs completed successfully!"
else
    echo "⚠️  One or more GPUs failed. Check logs:"
    echo "   $LOG_DIR/gpu0.log"
    echo "   $LOG_DIR/gpu1.log"
fi

echo ""
echo "Results:"
echo "  optimization_results_26personas/gpu0/"
echo "  optimization_results_26personas/gpu1/"
echo ""
echo "Logs:"
echo "  $LOG_DIR/"
echo "================================================================================"
