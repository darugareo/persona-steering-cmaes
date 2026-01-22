#!/bin/bash

# Phase 2: σ=2.5 re-optimization on GPU 1
# 7 high-L2 personas

env CUDA_VISIBLE_DEVICES=1 python scripts/reoptimize_sigma2_5.py \
    --output_dir optimization_results_sigma2.5 \
    --sigma 2.5 \
    --personas episode-19493_A episode-158821_B episode-16276_B \
               episode-24275_A episode-74475_A episode-184019_A \
               episode-128744_B \
    2>&1 | tee phase2_sigma25_gpu1.log
