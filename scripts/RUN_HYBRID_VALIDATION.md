# 大規模Hybrid検証実験 - 実行ガイド

**作成日**: 2026-01-04

---

## 実験の目的

Hybrid (Prompt + Steering) vs Prompt-only を大規模に比較し、統計的に有意な証拠を得る。

### 解決する3つの問題

1. **Train/Test Split**: v2の28プロンプト（最適化に未使用）でテスト
2. **Judge Overfitting**: GPT-4o使用（最適化時のGPT-4o-miniと異なる）
3. **Sample Size**: 728比較（既存の10比較から大幅拡大）

---

## 実験設計

### データ
- **ペルソナ**: 26個（全て最適化済み）
- **プロンプト**: v2の28個（最適化に未使用）
- **総比較数**: 26 × 28 = **728比較**

### 条件
1. **Hybrid**: ペルソナプロンプト + 最適化済みステアリング
2. **Prompt-only**: ペルソナプロンプトのみ

### Judge
- **モデル**: GPT-4o（最適化時のGPT-4o-miniと異なる）
- **形式**: ペアワイズ比較
- **位置ランダム化**: A/B順序をランダム化して位置バイアスを回避

---

## 実行方法

### 基本実行（フル実験）

```bash
# GPU 0で実行
CUDA_VISIBLE_DEVICES=0 python scripts/validate_hybrid_large_scale.py --gpu_id 0

# GPU 1で実行
CUDA_VISIBLE_DEVICES=1 python scripts/validate_hybrid_large_scale.py --gpu_id 1
```

### テスト実行（少数サンプル）

```bash
# 2ペルソナ × 3プロンプト = 6比較でテスト
python scripts/validate_hybrid_large_scale.py \
    --gpu_id 0 \
    --limit-personas 2 \
    --limit-prompts 3
```

### カスタマイズオプション

```bash
python scripts/validate_hybrid_large_scale.py \
    --gpu_id 0 \
    --layer 20 \              # ステアリング層（デフォルト: 20）
    --alpha 2.0 \             # ステアリング強度（デフォルト: 2.0）
    --judge-model gpt-4o \    # Judgeモデル（デフォルト: gpt-4o）
    --resume results/hybrid_validation_large_scale/comparison_results.json  # 中断から再開
```

### 中断からの再開

```bash
# 実験が中断した場合、checkpointから再開可能
python scripts/validate_hybrid_large_scale.py \
    --gpu_id 0 \
    --resume results/hybrid_validation_large_scale/comparison_results.json
```

---

## 出力ファイル

### 1. comparison_results.json
**パス**: `results/hybrid_validation_large_scale/comparison_results.json`

全728比較の詳細データ:
```json
{
  "date": "2026-01-04T...",
  "config": {
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "layer": 20,
    "alpha": 2.0,
    "judge_model": "gpt-4o"
  },
  "total_comparisons": 728,
  "results": [
    {
      "persona_id": "episode-184019_A",
      "prompt": "What's your opinion on...",
      "response_hybrid": "...",
      "response_prompt": "...",
      "winner": "hybrid",
      "confidence": 5,
      "explanation": "...",
      "hybrid_is_a": true,
      "judge_winner": "A"
    },
    ...
  ]
}
```

### 2. summary.json
**パス**: `results/hybrid_validation_large_scale/summary.json`

統計分析サマリー:
```json
{
  "date": "2026-01-04T...",
  "total_comparisons": 728,
  "wins": {
    "hybrid": 450,
    "prompt": 250,
    "ties": 28
  },
  "win_rates": {
    "hybrid": 0.618,
    "prompt": 0.343,
    "tie": 0.038
  },
  "statistical_tests": {
    "binomial_test_p_value": 0.0001,
    "significant_at_0.05": true,
    "cohens_h": 0.56,
    "effect_size_interpretation": "medium"
  },
  "confidence_interval_95": {
    "hybrid_win_rate_lower": 0.580,
    "hybrid_win_rate_upper": 0.655
  },
  "average_judge_confidence": 4.2,
  "per_persona": {
    "episode-184019_A": {"hybrid": 18, "prompt": 8, "tie": 2},
    ...
  }
}
```

---

## 統計分析

### 1. 二項検定（Binomial Test）
**帰無仮説**: Hybrid勝率 = 50%
**対立仮説**: Hybrid勝率 ≠ 50%

```python
p_value = stats.binom_test(wins_hybrid, n=total, p=0.5)
```

**判定**:
- p < 0.05 → 有意差あり（Hybridが50%と異なる）
- p ≥ 0.05 → 有意差なし

### 2. 効果量（Cohen's h）
**定義**: 2つの比率の差を標準化した指標

```python
h = 2 * (arcsin(sqrt(p_hybrid)) - arcsin(sqrt(p_prompt)))
```

**解釈**:
- |h| < 0.2: negligible（無視できる）
- |h| < 0.5: small（小）
- |h| < 0.8: medium（中）
- |h| ≥ 0.8: large（大）

### 3. Wilson Score 信頼区間
**95%信頼区間**: Hybrid勝率の真の値の推定範囲

正規近似より正確（特にサンプルサイズが小~中程度の場合）

---

## 推定実行時間・コスト

### 時間
- **生成**: 728 × 2条件 = 1,456回
  - GPU推論: ~3秒/回
  - 総生成時間: ~1.2時間

- **Judge評価**: 728回
  - GPT-4o API: ~2秒/回
  - 総評価時間: ~0.4時間

**合計**: 約1.5-2時間

### コスト
- **GPT-4o API**: 728回
  - 入力: ~500 tokens/回 × 728 = 364,000 tokens
  - 出力: ~200 tokens/回 × 728 = 145,600 tokens
  - 推定コスト: ~$10-15

---

## 期待される結果

### ベストケース
```
Hybrid勝率: 65-75%
p-value: < 0.001
Cohen's h: 0.6-0.8 (medium-large)
```

**解釈**: Hybridが統計的に有意に優位 → 論文の主張を強力に支持

### ワーストケース
```
Hybrid勝率: 45-55%
p-value: > 0.05
Cohen's h: 0.0-0.2 (negligible)
```

**解釈**: 有意差なし → 手法の再検討が必要

---

## トラブルシューティング

### エラー: "Optimized weights not found"
**原因**: ペルソナの重みファイルが見つからない

**解決策**:
```bash
# 最適化済みペルソナリストを確認
cat optimization_results_26personas/analysis_summary.json | grep persona_id
```

### エラー: "CUDA out of memory"
**原因**: GPU メモリ不足

**解決策**:
```bash
# より小さいバッチで実行（自動的に1個ずつ処理）
# または別のGPUを使用
CUDA_VISIBLE_DEVICES=1 python scripts/validate_hybrid_large_scale.py --gpu_id 1
```

### エラー: "OpenAI API rate limit"
**原因**: API呼び出し制限

**解決策**:
```python
# persona_judge_evaluator.py に自動リトライロジックが実装済み
# 自動的に待機して再試行されます
```

### 実験中断時
**解決策**:
```bash
# checkpointから再開
python scripts/validate_hybrid_large_scale.py \
    --resume results/hybrid_validation_large_scale/comparison_results.json
```

---

## 次のステップ

### 実験完了後

1. **結果の確認**:
```bash
cat results/hybrid_validation_large_scale/summary.json
```

2. **詳細分析**:
```python
import json
with open('results/hybrid_validation_large_scale/comparison_results.json') as f:
    data = json.load(f)

# ペルソナごとの分析
# Confidenceスコア分布
# 失敗ケースの調査
```

3. **論文への組み込み**:
- 統計的有意性を報告
- 効果量を報告
- 95%信頼区間を報告

---

## チェックリスト

実行前:
- [ ] OpenAI API keyが設定されている（.envまたは環境変数）
- [ ] GPUが利用可能（`nvidia-smi`で確認）
- [ ] 最適化済み重みが存在（`optimization_results_26personas/`）
- [ ] v2プロンプトが存在（`data/eval_prompts/persona_eval_prompts_v2.json`）

実行中:
- [ ] 進捗が表示される（tqdmプログレスバー）
- [ ] checkpointが10比較ごとに保存される
- [ ] エラーが発生しても継続する

実行後:
- [ ] `comparison_results.json`が生成された（728行）
- [ ] `summary.json`が生成された
- [ ] 統計的検定結果が表示された

---

**重要**: この実験は論文投稿に必須です。Train/Test Splitがない状態では査読を通過できません。
