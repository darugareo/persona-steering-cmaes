#!/bin/bash
# Monitor 2-trait optimization and stop after 2 hours if not complete

TIMEOUT_SECONDS=7200  # 2 hours
START_TIME=$(date +%s)
DEADLINE=$((START_TIME + TIMEOUT_SECONDS))

echo "監視開始: $(date)"
echo "制限時間: 2時間 (終了予定: $(date -d @$DEADLINE))"
echo ""

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    REMAINING=$((DEADLINE - CURRENT_TIME))

    # Check if all optimizations are complete
    COMPLETE_COUNT=0
    for log in logs/two_trait_optimization/episode-*.log; do
        if grep -q "Optimization complete!" "$log" 2>/dev/null; then
            COMPLETE_COUNT=$((COMPLETE_COUNT + 1))
        fi
    done

    # Check current generations
    echo "=== $(date) - 経過時間: $((ELAPSED/60))分 / 残り時間: $((REMAINING/60))分 ==="
    for log in logs/two_trait_optimization/episode-*.log; do
        PERSONA=$(basename "$log" .log)
        GEN=$(grep "Generation " "$log" 2>/dev/null | tail -1 || echo "起動中...")
        echo "  $PERSONA: $GEN"
    done
    echo "  完了: $COMPLETE_COUNT/4"
    echo ""

    # If all complete, exit successfully
    if [ $COMPLETE_COUNT -eq 4 ]; then
        echo "✅ 全ての最適化が完了しました！"
        exit 0
    fi

    # If timeout reached, kill processes
    if [ $CURRENT_TIME -ge $DEADLINE ]; then
        echo "⏰ 制限時間（2時間）に達しました。最適化を中止します。"
        echo ""

        # Kill all two_trait_optimizer processes
        PIDS=$(ps aux | grep two_trait_optimizer | grep -v grep | awk '{print $2}')
        if [ -n "$PIDS" ]; then
            echo "以下のプロセスを終了します:"
            ps aux | grep two_trait_optimizer | grep -v grep
            echo ""
            echo "$PIDS" | xargs kill -9
            echo "✅ プロセスを終了しました"
        else
            echo "実行中のプロセスが見つかりません"
        fi

        # Save partial results
        echo ""
        echo "部分的な結果を保存します..."
        mkdir -p optimization_results_2trait_partial
        for persona in episode-184019_A episode-118328_B episode-239427_A episode-225888_A; do
            if [ -f "logs/two_trait_optimization/${persona}.log" ]; then
                cp "logs/two_trait_optimization/${persona}.log" "optimization_results_2trait_partial/${persona}_partial.log"
            fi
        done
        echo "✅ ログを optimization_results_2trait_partial/ に保存しました"

        exit 1
    fi

    # Wait 5 minutes before next check
    sleep 300
done
