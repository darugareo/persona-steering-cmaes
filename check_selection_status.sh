#!/bin/bash
# ペルソナ特有ターン選定の進捗確認

echo "==================================="
echo "ペルソナ特有ターン選定 進捗確認"
echo "==================================="
echo ""

# プロセス確認
if ps aux | grep "select_persona_specific_turns.py" | grep -v grep > /dev/null; then
    echo "✅ 処理実行中"
    COMPLETED=$(grep -c "^\[" persona_selection.log)
    echo "   進捗: $COMPLETED/28 ペルソナ完了"
    REMAINING=$((28 - COMPLETED))
    EST_MIN=$((REMAINING * 50 / 60))
    echo "   推定残り時間: 約${EST_MIN}分"
else
    echo "✅ 処理完了"
fi

echo ""
echo "--- 最新の結果 ---"
grep "Results:" persona_selection.log | tail -n 3

echo ""
echo "--- エラー確認 ---"
if grep -i "error\|exception" persona_selection.log > /dev/null; then
    echo "⚠️ エラーが検出されました"
    grep -i "error\|exception" persona_selection.log | tail -n 5
else
    echo "✅ エラーなし"
fi

echo ""
echo "==================================="
echo "詳細ログ: tail -f persona_selection.log"
echo "==================================="
