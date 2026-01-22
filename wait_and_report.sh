#!/bin/bash
# 処理完了を待って報告書を生成

echo "ペルソナ特有ターン選定の完了を待機中..."

while ps aux | grep "select_persona_specific_turns.py" | grep -v grep > /dev/null; do
    COMPLETED=$(grep -c "^\[" persona_selection.log 2>/dev/null || echo "0")
    echo "[$(date +%H:%M:%S)] 進捗: $COMPLETED/28 ペルソナ完了"
    sleep 60
done

echo ""
echo "✅ 処理完了！報告書を生成します..."
sleep 2

# 結果の集計
python3 scripts/summarize_persona_selection.py

echo "✅ 報告書生成完了"
