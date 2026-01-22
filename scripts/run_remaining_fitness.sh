#!/bin/bash
# Run remaining fitness optimizations sequentially on GPU 0

cd /data01/nakata/master_thesis/persona2

# Wait for bertscore to complete
echo "Waiting for ep239427_bertscore to complete..."
wait 84160

# Run style
echo "Starting ep239427_style..."
CUDA_VISIBLE_DEVICES=0 python scripts/fitness_comparison_optimizer.py \
  --persona_id episode-239427_A \
  --fitness_type style \
  --gpu_id 0 \
  --max_generations 10 \
  > logs/fitness_comparison/ep239427_style_retry.log 2>&1

# Run judge
echo "Starting ep239427_judge..."
CUDA_VISIBLE_DEVICES=0 python scripts/fitness_comparison_optimizer.py \
  --persona_id episode-239427_A \
  --fitness_type judge \
  --gpu_id 0 \
  --max_generations 10 \
  > logs/fitness_comparison/ep239427_judge_retry.log 2>&1

# Run combined
echo "Starting ep239427_combined..."
CUDA_VISIBLE_DEVICES=0 python scripts/fitness_comparison_optimizer.py \
  --persona_id episode-239427_A \
  --fitness_type combined \
  --gpu_id 0 \
  --max_generations 10 \
  > logs/fitness_comparison/ep239427_combined_retry.log 2>&1

echo "All optimizations complete!"
date
