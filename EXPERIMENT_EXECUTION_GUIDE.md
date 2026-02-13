# 実験実行ガイド（論文化用・最小セット）

**作成日**: 2026-01-27
**目的**: 論文化のための3つの追実験を実行する
**予想APIコスト**: $14以内（目標$100に対して安全マージン）

---

## 📋 実験概要

### 実験① Trait Shuffle Ablation（最優先）
- **目的**: Steering効果が「重みの大きさ」ではなく意味のある方向構造に依存することを示す
- **対象**: 4ペルソナ（効果確認済み）
- **サンプル数**: 240評価（4ペルソナ × 20プロンプト × 3比較）
- **推定コスト**: $1.80
- **推定時間**: 3-4時間（生成2時間 + 評価1-2時間）

### 実験② Layer Shift Ablation
- **目的**: 層選択が恣意的でないことの確認、Cross-model転移の効果減衰説明
- **対象**: 4ペルソナ
- **サンプル数**: 480評価（4ペルソナ × 20プロンプト × 3層 × 2比較）
- **推定コスト**: $3.60
- **推定時間**: 4-5時間（生成3時間 + 評価1-2時間）

### 実験③ 10ペルソナ完全評価（Judge感度改善）
- **目的**: 既存10ペルソナ結果のtie率問題を解消、定量結果を安定化
- **対象**: 10ペルソナ
- **サンプル数**: 840評価（10ペルソナ × 28プロンプト × 3比較）
- **推定コスト**: $6.30
- **推定時間**: 8-10時間（生成6時間 + 評価2-4時間）

**総コスト**: $11.70
**総時間**: 15-19時間

---

## 🔧 事前準備

### 1. 環境確認

```bash
cd /data01/nakata/master_thesis/persona2

# 必要なデータが揃っているか確認
ls data/steering_vectors_v2/R1/layer20_svd.pt
ls optimization_results/episode-184019_A/best_weights.json

# すべて存在すればOK
```

### 2. OpenAI APIキー設定

```bash
# .envファイルを作成（まだなければ）
cp .env.example .env

# .envを編集してAPIキーを設定
nano .env
```

**.env の内容**:
```bash
OPENAI_API_KEY=your_actual_openai_api_key_here
```

**重要**: Judge modelは `gpt-4o`（miniではない）を使用します。

### 3. GPU確認

```bash
# GPU使用状況を確認
nvidia-smi

# Llama-3-8Bには16GB以上のVRAM推奨
```

---

## 🚀 実行手順

### 実験① Trait Shuffle Ablation

```bash
# tmuxセッション開始（推奨）
tmux new -s trait_shuffle

# 実験実行
python scripts/run_trait_shuffle_ablation.py

# tmuxデタッチ: Ctrl+B → D
```

**進捗確認**:
```bash
# リアルタイムで確認
tmux attach -t trait_shuffle

# 結果ファイルで確認
ls results/trait_shuffle/*/summary.json
```

**完了条件**:
- 4ペルソナすべてのsummary.jsonが生成
- `results/trait_shuffle/aggregate_summary.json` 存在
- normal > shuffled が有意に多い
- shuffled ≈ base 程度

**想定出力**:
```
results/trait_shuffle/
├── episode-184019_A/
│   ├── generations.json
│   ├── normal_vs_shuffled.json
│   ├── normal_vs_base.json
│   └── summary.json
├── episode-118328_B/
│   └── ...
├── episode-239427_A/
│   └── ...
├── episode-225888_A/
│   └── ...
└── aggregate_summary.json
```

---

### 実験② Layer Shift Ablation

```bash
# tmuxセッション開始
tmux new -s layer_shift

# 実験実行
python scripts/run_layer_shift_ablation.py

# tmuxデタッチ: Ctrl+B → D
```

**進捗確認**:
```bash
tmux attach -t layer_shift

# または結果確認
ls results/layer_shift/*/summary.json
```

**完了条件**:
- 4ペルソナすべてのsummary.jsonが生成
- L_opt（Layer 20）が最も高い勝率
- L_opt±5 で性能低下またはtie増加

**想定出力**:
```
results/layer_shift/
├── episode-184019_A/
│   ├── generations.json
│   ├── layer_opt_vs_minus5.json
│   ├── layer_opt_vs_plus5.json
│   └── summary.json
├── ...
└── aggregate_summary.json
```

---

### 実験③ 10ペルソナ完全評価

```bash
# tmuxセッション開始
tmux new -s eval_10personas

# 実験実行
python scripts/run_10personas_complete_evaluation.py

# tmuxデタッチ: Ctrl+B → D
```

**進捗確認**:
```bash
tmux attach -t eval_10personas

# 結果確認
ls results/10personas_gpt4o/*/summary.json
```

**完了条件**:
- 10ペルソナすべてのsummary.jsonが生成
- tie率が70%未満に低下
- Optimized vs Base が p < 0.001

**想定出力**:
```
results/10personas_gpt4o/
├── episode-184019_A/
│   ├── generations.json
│   ├── base_vs_prompt.json
│   ├── base_vs_equal.json
│   ├── base_vs_optimized.json
│   └── summary.json
├── ...（10ペルソナ分）
└── aggregate_summary.json
```

---

## 📊 結果の確認

### 実験①の成功基準

```bash
# Aggregate summaryを確認
cat results/trait_shuffle/aggregate_summary.json | jq '.[] | {persona: .persona_id, normal_vs_shuffled: .normal_vs_shuffled.normal_win_rate}'
```

**期待値**:
- Normal win rate > 60% for all personas
- Shuffled ≈ Base (shuffle効果なし)

### 実験②の成功基準

```bash
cat results/layer_shift/aggregate_summary.json | jq '.[] | {persona: .persona_id, L_opt_minus: .L_opt_vs_L_minus.L_opt_win_rate, L_opt_plus: .L_opt_vs_L_plus.L_opt_win_rate}'
```

**期待値**:
- L_opt win rate > 50% vs L_minus
- L_opt win rate > 50% vs L_plus

### 実験③の成功基準

```bash
cat results/10personas_gpt4o/aggregate_summary.json | jq '.[] | {persona: .persona_id, tie_rate_opt: .base_vs_optimized.tie_rate}'
```

**期待値**:
- Average tie rate < 70%（改善目標）
- Optimized win rate > Base（有意差あり）

---

## 🔍 トラブルシューティング

### GPUメモリ不足

```bash
# エラー: CUDA out of memory

# 対処法: bfloat16を使用（既に設定済み）
# または、他のプロセスを終了
nvidia-smi
kill -9 <PID>
```

### API Rate Limit

```bash
# エラー: Rate limit exceeded

# 対処法: スクリプトを中断して5-10分待機してから再開
# OpenAI API Tier 1: 500 RPM, Tier 2: 5000 RPM
```

### Judge評価エラー

```bash
# エラー: JSON parse error in judge

# 対処法: スクリプトは自動的にtieとしてフォールバック
# 手動で修正する場合は各persona/の*.jsonを確認
```

---

## 📈 論文への統合

### 実験①の使用箇所

**Results Section (Ablation Study)**:
> "To verify that steering effectiveness depends on meaningful direction structure rather than weight magnitude alone, we conducted a trait shuffle ablation. We randomly permuted trait dimensions while preserving L2 norm. Results show that normal vectors significantly outperform shuffled vectors (win rate: X.X% vs Y.Y%, p<0.05), while shuffled vectors perform similarly to baseline, confirming that semantic direction is critical."

### 実験②の使用箇所

**Results Section (Layer Sensitivity)**:
> "Layer selection analysis reveals that steering effectiveness is layer-dependent. Applying the optimized vector at layer 20 (L_opt) achieves superior performance compared to layers 15 (L_opt-5) or 25 (L_opt+5), with win rates of X.X% and Y.Y% respectively. This layer dependency explains the reduced effectiveness observed in cross-model transfer."

### 実験③の使用箇所

**Results Section (Main Quantitative Results)**:
> "Evaluation across 10 personas shows that the optimized steering method significantly outperforms the baseline (win rate: X.X%, p<0.001), as well as equal-weight steering (win rate: Y.Y%, p<0.001). Using GPT-4o as judge reduced tie rate to Z.Z% compared to GPT-4o-mini (previously 77-88%), enabling more reliable discrimination."

---

## ✅ 実行チェックリスト

**実験開始前**:
- [ ] SVD vectors存在確認（`data/steering_vectors_v2/`）
- [ ] Optimization results存在確認（`optimization_results/`）
- [ ] OpenAI API key設定（`.env`）
- [ ] GPU利用可能確認（`nvidia-smi`）

**実験①**:
- [ ] Script実行完了
- [ ] 4ペルソナすべて成功
- [ ] Aggregate summary生成
- [ ] Normal > Shuffled確認

**実験②**:
- [ ] Script実行完了
- [ ] 4ペルソナすべて成功
- [ ] L_opt優位性確認

**実験③**:
- [ ] Script実行完了
- [ ] 10ペルソナすべて成功
- [ ] Tie率改善確認
- [ ] 統計的有意性確認

---

## 📞 次のステップ

実験完了後:

1. **結果の統計分析**
   ```bash
   python scripts/statistical_analysis_paper.py
   ```

2. **論文用テーブル生成**
   ```bash
   python scripts/generate_paper_tables.py
   ```

3. **LaTeXテーブル統合**
   - Tables → Paper draft

---

**注意事項**:
- すべてのスクリプトは自動保存・エラーハンドリング実装済み
- 途中で中断しても、既存結果は保持される
- 再実行時は既存ファイルを上書き
- tmux使用推奨（SSH切断対策）

**問題発生時の連絡先**:
- GitHub Issues: https://github.com/anthropics/claude-code/issues
- プロジェクトREADME参照
