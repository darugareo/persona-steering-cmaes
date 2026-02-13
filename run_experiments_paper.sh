#!/bin/bash
#
# 論文化用実験実行スクリプト
# 3つの実験を順番に実行
#

set -e  # エラー時に停止

echo "========================================"
echo "論文化用実験実行スクリプト"
echo "========================================"
echo ""

# 環境チェック
echo "環境チェック中..."

# OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
    else
        echo "ERROR: OPENAI_API_KEY が設定されていません"
        echo "  .envファイルを作成してAPIキーを設定してください"
        exit 1
    fi
fi

# SVD vectors
if [ ! -f "data/steering_vectors_v2/R1/layer20_svd.pt" ]; then
    echo "ERROR: SVD vectors が見つかりません"
    echo "  data/steering_vectors_v2/ を確認してください"
    exit 1
fi

# Optimization results
if [ ! -f "optimization_results/episode-184019_A/best_weights.json" ]; then
    echo "ERROR: Optimization results が見つかりません"
    echo "  optimization_results/ を確認してください"
    exit 1
fi

echo "✓ 環境チェック完了"
echo ""

# 実験選択
echo "実行する実験を選択してください:"
echo "  1) 実験① Trait Shuffle Ablation のみ"
echo "  2) 実験② Layer Shift Ablation のみ"
echo "  3) 実験③ 10ペルソナ完全評価 のみ"
echo "  4) すべての実験を順番に実行（推奨）"
echo "  5) カスタム（対話形式で選択）"
echo ""

if [ -z "$1" ]; then
    read -p "選択 (1-5): " choice
else
    choice=$1
fi

echo ""
echo "========================================"

# 実験実行
case $choice in
    1)
        echo "実験① Trait Shuffle Ablation を実行"
        echo "========================================"
        python scripts/run_trait_shuffle_ablation.py
        ;;
    2)
        echo "実験② Layer Shift Ablation を実行"
        echo "========================================"
        python scripts/run_layer_shift_ablation.py
        ;;
    3)
        echo "実験③ 10ペルソナ完全評価 を実行"
        echo "========================================"
        python scripts/run_10personas_complete_evaluation.py
        ;;
    4)
        echo "すべての実験を順番に実行"
        echo "========================================"
        echo ""

        echo "[1/3] 実験① Trait Shuffle Ablation"
        python scripts/run_trait_shuffle_ablation.py
        echo ""
        echo "✓ 実験①完了"
        echo ""

        echo "[2/3] 実験② Layer Shift Ablation"
        python scripts/run_layer_shift_ablation.py
        echo ""
        echo "✓ 実験②完了"
        echo ""

        echo "[3/3] 実験③ 10ペルソナ完全評価"
        python scripts/run_10personas_complete_evaluation.py
        echo ""
        echo "✓ 実験③完了"
        ;;
    5)
        echo "カスタム実行"
        echo "========================================"

        read -p "実験①を実行しますか？ (y/n): " exp1
        if [ "$exp1" = "y" ]; then
            python scripts/run_trait_shuffle_ablation.py
            echo "✓ 実験①完了"
        fi

        read -p "実験②を実行しますか？ (y/n): " exp2
        if [ "$exp2" = "y" ]; then
            python scripts/run_layer_shift_ablation.py
            echo "✓ 実験②完了"
        fi

        read -p "実験③を実行しますか？ (y/n): " exp3
        if [ "$exp3" = "y" ]; then
            python scripts/run_10personas_complete_evaluation.py
            echo "✓ 実験③完了"
        fi
        ;;
    *)
        echo "無効な選択です"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "実験実行完了！"
echo "========================================"
echo ""
echo "結果ディレクトリ:"
echo "  - results/trait_shuffle/"
echo "  - results/layer_shift/"
echo "  - results/10personas_gpt4o/"
echo ""
echo "詳細は EXPERIMENT_EXECUTION_GUIDE.md を参照してください"
echo ""
