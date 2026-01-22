#!/bin/bash
#
# σ=5.0で全ペルソナを最適化（GPU 1のみ使用）
# GPU 0はα×σ実験用に空けておく
#

OUTPUT_DIR="optimization_results_sigma5"
SIGMA=5.0
ALPHA=2.0
GPU=1

# 全ペルソナリスト
PERSONAS=(
    "episode-118328_B"
    "episode-128744_B"
    "episode-134226_A"
    "episode-136981_B"
    "episode-137872_B"
    "episode-140544_B"
    "episode-14330_A"
    "episode-145896_A"
    "episode-158821_B"
    "episode-16276_B"
    "episode-175246_A"
    "episode-179307_A"
    "episode-184019_A"
    "episode-19493_A"
    "episode-204347_A"
    "episode-223194_B"
    "episode-225888_A"
    "episode-239427_A"
    "episode-24275_A"
    "episode-36796_A"
    "episode-36796_B"
    "episode-37624_A"
    "episode-38144_A"
    "episode-51953_A"
    "episode-74475_A"
    "episode-84804_A"
    "episode-98323_A"
    "episode-98947_A"
)

echo "================================================================================"
echo "σ=5.0 SEQUENTIAL OPTIMIZATION (GPU 1 only)"
echo "================================================================================"
echo "  Total personas: ${#PERSONAS[@]}"
echo "  Using: GPU $GPU"
echo "  Output: $OUTPUT_DIR"
echo "  Estimated time: ~${#PERSONAS[@]} x 30min = ~$((${#PERSONAS[@]}*30/60)) hours"
echo "================================================================================"
echo ""

START_TIME=$(date +%s)

for i in "${!PERSONAS[@]}"; do
    persona="${PERSONAS[$i]}"
    log_file="${OUTPUT_DIR}/${persona}/optimization_log.txt"

    mkdir -p "${OUTPUT_DIR}/${persona}"

    echo ""
    echo "================================================================================"
    echo "[$((i+1))/${#PERSONAS[@]}] $persona"
    echo "================================================================================"

    CUDA_VISIBLE_DEVICES=$GPU python scripts/optimize_single_persona_sigma5.py \
        --persona_id "$persona" \
        --device "cuda:0" \
        --output_dir "$OUTPUT_DIR" \
        --sigma "$SIGMA" \
        --alpha "$ALPHA" \
        --max_iterations 20 \
        --population_size 10 \
        2>&1 | tee "$log_file"

    if [ $? -eq 0 ]; then
        echo "✅ Completed: $persona"
    else
        echo "❌ Failed: $persona"
    fi
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "================================================================================"
echo "✅ ALL OPTIMIZATIONS COMPLETE"
echo "================================================================================"
echo "  Duration: $((DURATION/3600))h $((DURATION%3600/60))m"
echo "================================================================================"
echo ""

# 結果を集計
echo "Aggregating results..."
python -c "
import json
import numpy as np
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
personas = $( IFS=,; echo "[$(printf '\"%s\",' "${PERSONAS[@]}" | sed 's/,$//')]" )

all_results = []
for persona_id in personas:
    summary_path = output_dir / persona_id / 'summary.json'
    if summary_path.exists():
        with open(summary_path) as f:
            all_results.append(json.load(f))

if all_results:
    l2_norms = [r['l2_norm'] for r in all_results]

    print()
    print('='*80)
    print('L2 NORM STATISTICS')
    print('='*80)
    print(f'  Count: {len(l2_norms)}')
    print(f'  Mean: {np.mean(l2_norms):.3f}')
    print(f'  Std: {np.std(l2_norms):.3f}')
    print(f'  Min: {np.min(l2_norms):.3f}')
    print(f'  Max: {np.max(l2_norms):.3f}')
    print(f'  Median: {np.median(l2_norms):.3f}')
    print()
    print('Distribution:')
    print(f'  < 5.0: {sum(1 for x in l2_norms if x < 5.0)}')
    print(f'  5.0-7.0: {sum(1 for x in l2_norms if 5.0 <= x < 7.0)}')
    print(f'  7.0-10.0: {sum(1 for x in l2_norms if 7.0 <= x < 10.0)}')
    print(f'  >= 10.0: {sum(1 for x in l2_norms if x >= 10.0)}')
    print('='*80)

    # Save aggregate
    with open(output_dir / 'all_results_summary.json', 'w') as f:
        json.dump({
            'total': len(all_results),
            'sigma': $SIGMA,
            'alpha': $ALPHA,
            'l2_stats': {
                'mean': float(np.mean(l2_norms)),
                'std': float(np.std(l2_norms)),
                'min': float(np.min(l2_norms)),
                'max': float(np.max(l2_norms)),
                'median': float(np.median(l2_norms))
            },
            'results': all_results
        }, f, indent=2)

    print(f'Saved: {output_dir}/all_results_summary.json')
"
