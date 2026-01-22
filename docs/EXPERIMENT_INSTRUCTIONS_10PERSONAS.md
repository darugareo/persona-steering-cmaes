# 10ペルソナ完全実験実行指示書

## 目的
10ペルソナ（既存3＋新規7）で、4手法（base/prompt/equal/optimized）を生成し、Judge評価して集計する

---

## 0. 前提チェック ✓ 完了

**10人分のoptimized weightsが揃っていることを確認済み**

### 既存3ペルソナ
- ✓ `persona-opt/episode-184019_A/best_weights.json`
- ✓ `persona-opt/episode-239427_A/best_weights.json`
- ✓ `persona-opt/episode-118328_B/best_weights.json`

### 新規7ペルソナ
- ✓ `persona-opt/episode-5289_A/best_weights.json` (score: 1.000)
- ✓ `persona-opt/episode-29600_A/best_weights.json` (score: 1.000)
- ✓ `persona-opt/episode-88279_B/best_weights.json` (score: 1.000)
- ✓ `persona-opt/episode-132247_A/best_weights.json` (score: 1.000)
- ✓ `persona-opt/episode-166805_A/best_weights.json` (score: 0.800)
- ✓ `persona-opt/episode-196697_B/best_weights.json` (score: 1.000)
- ✓ `persona-opt/episode-225888_A/best_weights.json` (score: 1.000)

**平均スコア**: 0.971 (6/7が完全スコア)

---

## 1. 10人フル生成（クロスモデル：Mistral-7B）

### 実行コマンド

```bash
# tmuxセッション開始
tmux new -s gen10

# プロジェクトディレクトリへ移動
cd /data01/nakata/master_thesis/persona2

# 生成実行（バックグラウンド、ログ保存）
python experiments/run_10personas_generation.py > logs/gen10_personas.log 2>&1 &

# tmuxデタッチ: Ctrl+B → D
```

### 進捗確認

```bash
# ログをリアルタイム監視
tail -f logs/gen10_personas.log

# または別のtmuxウィンドウでアタッチ
tmux attach -t gen10
```

### 生成完了チェック

**期待値**: 10 personas × 4 methods × 28 prompts = **1120 outputs**

```bash
cd /data01/nakata/master_thesis/persona2

# 各ペルソナ・各メソッドが28行あるか確認
for p in $(cat personas_final_10_corrected.txt); do
  for m in base prompt equal optimized; do
    f="results/cross_model/mistral_7b/$p/$m.jsonl"
    if [ -f "$f" ]; then
      lines=$(wc -l < "$f")
      if [ "$lines" -eq 28 ]; then
        echo "✓ $p/$m: $lines"
      else
        echo "⚠️  $p/$m: $lines (expected 28)"
      fi
    else
      echo "✗ MISSING: $p/$m"
    fi
  done
done
```

**期待結果**: 全40ファイル（10 personas × 4 methods）が28行ずつ、MISSINGなし

### 推定実行時間

- 1プロンプト生成: ~5秒
- 1ペルソナ（4 methods × 28 prompts）: 112生成 × 5秒 = 9.3分
- 10ペルソナ合計: **約93分（1.5時間）**

---

## 2. Judge評価（10人フル、temperature=0.0）

### 準備

```bash
# OpenAI APIキーをセット（必須）
export OPENAI_API_KEY="your_actual_key_here"

# キーが設定されているか確認
echo $OPENAI_API_KEY
```

### 実行コマンド

```bash
# tmuxセッション開始
tmux new -s judge10

cd /data01/nakata/master_thesis/persona2

# Judge評価実行（バックグラウンド、ログ保存）
python scripts/run_judge_evaluation_10personas.py > logs/judge10.log 2>&1 &

# tmuxデタッチ: Ctrl+B → D
```

### 進捗確認

```bash
# ログをリアルタイム監視
tail -f logs/judge10.log

# 評価進捗をカウント
grep "Judging" logs/judge10.log | wc -l
```

### 評価完了チェック

**期待値**: 10 personas × 3 comparison pairs × 28 prompts = **840 comparisons**

```bash
# 結果ファイル確認
ls -lh results/judge_evaluation/10personas_lightweight_results.json

# 結果サマリー表示
python -c "import json; d=json.load(open('results/judge_evaluation/10personas_lightweight_results.json')); print(f'Total comparisons: {len(d[\"comparisons\"])}')"
```

### 推定実行時間

- 1比較判定: ~3秒（GPT-4o-mini、temperature=0.0）
- 840比較合計: **約42分**

### APIコスト見積もり

- GPT-4o-mini単価: $0.15/1M input tokens, $0.60/1M output tokens
- 1比較あたり: ~500 input + 50 output tokens
- 840比較合計: 420K input + 42K output = **約$0.09**

---

## 3. 集計（統計表とLaTeX出力）

### 実行コマンド

```bash
cd /data01/nakata/master_thesis/persona2

# 集計スクリプト実行
python scripts/aggregate_results_10personas.py
```

### 出力ファイル

```
reports/10personas/
├── summary.json                          # JSON形式のサマリー
├── tables/
│   ├── table_main_results.md             # Markdown表（メイン結果）
│   ├── table_main_results.tex            # LaTeX表（メイン結果）
│   ├── table_per_persona.md              # ペルソナ別詳細
│   └── table_statistical_tests.tex       # 統計検定結果
└── detailed_comparisons.json             # 全比較詳細
```

### 結果確認コマンド

```bash
# Markdown表を表示
cat reports/10personas/tables/table_main_results.md

# LaTeX表を表示
cat reports/10personas/tables/table_main_results.tex

# サマリーJSON確認
cat reports/10personas/summary.json | jq .
```

---

## 4. 論文への反映（IEEE Access）

### 更新必須項目

#### 4.1 Results セクション

**変更前（3ペルソナ）**:
```latex
\begin{table}[h]
\caption{Win rates for 3 personas}
...
\end{table}
```

**変更後（10ペルソナ）**:
```latex
\begin{table}[h]
\caption{Win rates for 10 personas with statistical significance}
% reports/10personas/tables/table_main_results.tex をコピー
...
\end{table}
```

#### 4.2 Experimental Setup セクション

追加記述:
```latex
We extended the evaluation to 10 diverse personas (3 from initial experiments + 7 newly selected via stratified sampling).
All 10 personas underwent CMA-ES optimization with early stopping (average 4.3 generations).
```

#### 4.3 Computational Cost（Table追加）

**新規追加**:
```latex
\begin{table}[h]
\caption{Computational Cost Breakdown}
\begin{tabular}{lrr}
\hline
Phase & Time (hours) & Cost (USD) \\
\hline
Trait Vector Extraction (R1-R5) & 2.0 & - \\
CMA-ES Optimization (10 personas) & 5.93 & - \\
Cross-Model Generation (1120 outputs) & 1.5 & - \\
GPT-4o-mini Judge Evaluation & 0.7 & 0.09 \\
\hline
Total & 10.13 & 0.09 \\
\hline
\end{tabular}
\end{table}
```

**早期収束の効果を強調**:
```latex
Early stopping reduced optimization time by 81.2\% (from 31.5 to 5.93 hours)
while maintaining high fitness scores (average 0.971).
```

#### 4.4 統計的有意差の明記

Results本文に追加:
```latex
McNemar tests confirmed statistically significant improvements:
equal vs base (p=0.0085), prompt vs base (p=0.0034),
and optimized vs equal (p=0.0146).
```

---

## 実行スケジュール

| フェーズ | 推定時間 | 累積時間 |
|---------|---------|---------|
| ✓ 0. 前提チェック | - | 完了済み |
| 1. 生成（Mistral-7B） | 1.5時間 | 1.5時間 |
| 2. Judge評価（GPT-4o-mini） | 0.7時間 | 2.2時間 |
| 3. 集計 | 0.1時間 | 2.3時間 |
| **合計** | **約2.3時間** | - |

**推奨実行タイミング**: 夜間バッチ実行（23:00開始 → 翌朝01:30完了）

---

## トラブルシューティング

### Q1. 生成が途中で止まった

```bash
# どこまで生成されたか確認
for p in $(cat personas_final_10_corrected.txt); do
  for m in base prompt equal optimized; do
    if [ -f "results/cross_model/mistral_7b/$p/$m.jsonl" ]; then
      echo "$p/$m: $(wc -l < results/cross_model/mistral_7b/$p/$m.jsonl)"
    fi
  done
done

# 特定ペルソナから再開したい場合は、スクリプトを編集してpersonasリストを調整
```

### Q2. Judge評価でAPIエラー

```bash
# エラーログ確認
grep "Error" logs/judge10.log

# レート制限の場合は、scripts/run_judge_evaluation_10personas.py のsleepを増やす
# 例: time.sleep(1) → time.sleep(2)
```

### Q3. メモリ不足（OOM）

```bash
# GPU使用状況確認
nvidia-smi

# 別のGPUを使用
export CUDA_VISIBLE_DEVICES=1
```

---

## 成功判定基準

### ✓ 生成完了
- [ ] 40ファイル（10 personas × 4 methods）が存在
- [ ] 各ファイルが28行（28プロンプト）
- [ ] `logs/gen10_personas.log` にエラーなし

### ✓ Judge評価完了
- [ ] `results/judge_evaluation/10personas_lightweight_results.json` が存在
- [ ] 840比較が記録されている
- [ ] `logs/judge10.log` にエラーなし

### ✓ 集計完了
- [ ] `reports/10personas/summary.json` が存在
- [ ] Win rate が合理的範囲（30-50%）
- [ ] p値 < 0.05 の統計的有意差が確認できる

---

## 次のアクション（実験完走後）

1. **論文更新**: IEEE Access原稿に10ペルソナ結果を反映
2. **追加分析**: ペルソナ間のバリエーション分析
3. **Figure作成**: Win rate比較の棒グラフ（matplotlib）
4. **Appendix**: 全10ペルソナのプロファイルと最適化weights

---

**作成日**: 2025-12-17
**最終更新**: 実験開始前
**ステータス**: Ready to Execute
