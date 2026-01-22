#!/bin/bash

# Phase 2: σ=2.5 re-optimization on GPU 0
# 7 high-L2 personas

env CUDA_VISIBLE_DEVICES=0 python scripts/reoptimize_sigma2_5.py \
    --output_dir optimization_results_sigma2.5 \
    --sigma 2.5 \
    --personas episode-179307_A episode-175246_A episode-134226_A \
               episode-136981_B episode-140544_B episode-118328_B \
               episode-84804_A \
    2>&1 | tee phase2_sigma25_gpu0.log
