#!/bin/bash
#
# σ=5.0で全ペルソナを並列最適化
# GPU 0とGPU 1を使用
#

OUTPUT_DIR="optimization_results_sigma5"
SIGMA=5.0
ALPHA=2.0

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
echo "σ=5.0 PARALLEL OPTIMIZATION"
echo "================================================================================"
echo "  Total personas: ${#PERSONAS[@]}"
echo "  Using: GPU 0 and GPU 1"
echo "  Output: $OUTPUT_DIR"
echo "================================================================================"
echo ""

# 関数: 1ペルソナを最適化
optimize_persona() {
    local persona=$1
    local gpu=$2
    local log_file="${OUTPUT_DIR}/${persona}/optimization_log.txt"

    mkdir -p "${OUTPUT_DIR}/${persona}"

    echo "[$(date '+%H:%M:%S')] Starting: $persona on cuda:$gpu"

    CUDA_VISIBLE_DEVICES=$gpu python scripts/optimize_single_persona_sigma5.py \
        --persona_id "$persona" \
        --device "cuda:0" \
        --output_dir "$OUTPUT_DIR" \
        --sigma "$SIGMA" \
        --alpha "$ALPHA" \
        --max_iterations 20 \
        --population_size 10 \
        > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✅ Completed: $persona"
    else
        echo "[$(date '+%H:%M:%S')] ❌ Failed: $persona"
    fi
}

export -f optimize_persona
export OUTPUT_DIR SIGMA ALPHA

# GNU Parallelを使用（インストール済みの場合）
if command -v parallel &> /dev/null; then
    echo "Using GNU Parallel for optimization..."

    # 2つのジョブを同時実行（GPU 0とGPU 1）
    printf "%s\n" "${PERSONAS[@]}" | parallel -j 2 --line-buffer optimize_persona {} {%}

else
    echo "GNU Parallel not found. Using sequential execution with background jobs..."

    # GPU 0とGPU 1に交互に割り当て
    for i in "${!PERSONAS[@]}"; do
        persona="${PERSONAS[$i]}"
        gpu=$((i % 2))

        # 2つまで並列実行
        while [ $(jobs -r | wc -l) -ge 2 ]; do
            sleep 10
        done

        optimize_persona "$persona" "$gpu" &
    done

    # 全ジョブの完了を待つ
    wait
fi

echo ""
echo "================================================================================"
echo "✅ ALL OPTIMIZATIONS COMPLETE"
echo "================================================================================"

# 結果を集計
python scripts/run_all_personas_sigma5_parallel.py --aggregate-only
