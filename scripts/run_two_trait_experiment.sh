#!/bin/bash
# Run 2-Trait Steering experiment for 4 effective personas

echo "================================================================================"
echo "2-Trait Steering Experiment (R2 + R4 only)"
echo "================================================================================"
echo ""
echo "Target personas (previously showed effect with 5-trait):"
echo "  1. episode-184019_A  (Husband, 75% steering win rate)"
echo "  2. episode-118328_B  (Wife, 60.7% steering win rate)"
echo "  3. episode-239427_A  (Neighbors, 42.9% steering win rate)"
echo "  4. episode-225888_A  (Classmates, 25% steering win rate)"
echo ""
echo "Settings:"
echo "  - Traits: R2, R4 only (2D optimization)"
echo "  - Generations: 20"
echo "  - Population: 8"
echo "  - Initial sigma: 2.0"
echo "  - Fitness: Style Similarity"
echo ""
echo "================================================================================"
echo ""

# Create log directory
mkdir -p logs/two_trait_optimization

# GPU 0: episode-184019_A and episode-239427_A
CUDA_VISIBLE_DEVICES=0 nohup python scripts/two_trait_optimizer.py \
    --persona_id episode-184019_A \
    --gpu_id 0 \
    --max_generations 20 \
    --popsize 8 \
    > logs/two_trait_optimization/episode-184019_A.log 2>&1 &
PID1=$!
echo "Started episode-184019_A on GPU 0 (PID: $PID1)"

sleep 5

CUDA_VISIBLE_DEVICES=0 nohup python scripts/two_trait_optimizer.py \
    --persona_id episode-239427_A \
    --gpu_id 0 \
    --max_generations 20 \
    --popsize 8 \
    > logs/two_trait_optimization/episode-239427_A.log 2>&1 &
PID2=$!
echo "Started episode-239427_A on GPU 0 (PID: $PID2)"

# GPU 1: episode-118328_B and episode-225888_A
CUDA_VISIBLE_DEVICES=1 nohup python scripts/two_trait_optimizer.py \
    --persona_id episode-118328_B \
    --gpu_id 0 \
    --max_generations 20 \
    --popsize 8 \
    > logs/two_trait_optimization/episode-118328_B.log 2>&1 &
PID3=$!
echo "Started episode-118328_B on GPU 1 (PID: $PID3)"

sleep 5

CUDA_VISIBLE_DEVICES=1 nohup python scripts/two_trait_optimizer.py \
    --persona_id episode-225888_A \
    --gpu_id 0 \
    --max_generations 20 \
    --popsize 8 \
    > logs/two_trait_optimization/episode-225888_A.log 2>&1 &
PID4=$!
echo "Started episode-225888_A on GPU 1 (PID: $PID4)"

echo ""
echo "All optimizations started!"
echo "  GPU 0: PID $PID1 (episode-184019_A), PID $PID2 (episode-239427_A)"
echo "  GPU 1: PID $PID3 (episode-118328_B), PID $PID4 (episode-225888_A)"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/two_trait_optimization/episode-184019_A.log"
echo "  tail -f logs/two_trait_optimization/episode-118328_B.log"
echo ""
echo "Check if running:"
echo "  ps aux | grep two_trait_optimizer"
echo ""
echo "Expected runtime: ~1-2 hours per persona"
echo "================================================================================"
