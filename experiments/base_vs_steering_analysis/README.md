# Base vs Steering Analysis Experiment

**実験日**: 2026-01-07
**目的**: なぜActivation Steeringが84.6%のpersonasで効果がなかったのかを解明

---

## クイックサマリー

**結論**:
- ✅ Steeringは**重みベクトルが大きい場合のみ**効果あり（L2ノルム > 5.0）
- ❌ 84.6%のpersonasでは重みが小さく（L2ノルム < 4）、効果なし
- 📊 効果あり群は効果なし群の**約2倍の重み**

---

## ファイル構成

```
experiments/base_vs_steering_analysis/
├── README.md                          # このファイル
├── REPORT.md                          # 詳細レポート（メイン）
├── analysis.ipynb                     # 分析用Jupyter notebook（実行済み）
├── weight_distributions.png           # 可視化1: Trait別重み分布
├── weight_l2_norm_comparison.png      # 可視化2: L2ノルム比較
├── weights_heatmap.png                # 可視化3: 重みヒートマップ
└── weight_effect_correlation.png      # 可視化4: 重み-効果相関
```

---

## 主要な数値

| 指標 | 値 |
|------|-----|
| 総personas | 26 |
| 効果なしpersonas | 22 (84.6%) |
| 効果ありpersonas | 4 (15.4%) |
| 総比較数 | 728 |
| 引き分け率 | 90.2% |
| 効果なし群 L2ノルム | 3.49 ± 1.54 |
| 効果あり群 L2ノルム | 7.18 ± 1.41 |
| **差** | **2.06倍** |

---

## 効果があった4 Personas

1. **episode-184019_A**: Steering勝率75%, L2ノルム7.61
2. **episode-118328_B**: Steering勝率60.7%, L2ノルム7.96
3. **episode-239427_A**: Steering勝率42.9%, L2ノルム5.06
4. **episode-225888_A**: Steering勝率25%, L2ノルム8.10

---

## 重要な発見

### 1. 重みの大きさが決定的
- L2ノルム > 5.0で効果が現れ始める
- L2ノルム < 4.0では全く効果なし

### 2. R4 (Trait 4) が重要
- 効果あり群のR4: -4.56 (p=0.0226, 有意)
- 効果なし群のR4: -1.75
- **R4が鍵となる特性**の可能性

### 3. Steeringの効果パターン
- より短い文
- カジュアルな表現（"me time", "big ol'"など）
- 会話的なトーン

### 4. 引き分けの理由
- **両方ともpersonaから逸脱**（58%）
- BaseもSteeringも効いていない

---

## 推奨される次のステップ

### 短期（すぐ実施可能）
1. ✅ **Alpha値を増やす**: α=2.0 → 5.0, 10.0, 20.0
2. ✅ **Layer sweepの実施**: Layer 10, 15, 20, 25で比較
3. ✅ **R4の意味的解釈**: Trait定義を確認

### 中期（数日〜数週間）
1. 🔧 **最適化手法の改善**: より大きな重みを許容
2. 🔧 **Persona固有のvectors**: 一般的なR1-R5ではなく個別最適化
3. 🔧 **評価指標の多様化**: Judge以外のmetrics追加

### 長期（研究方向）
1. 🔬 **新しいステアリング手法**: LoRA、Prompt tuningとの組み合わせ
2. 🔬 **理論的理解**: なぜ重みが小さく収束するのか

---

## 使い方

### レポートを読む
```bash
# 詳細レポート（推奨）
cat REPORT.md

# または
less REPORT.md
```

### ノートブックを開く
```bash
jupyter notebook analysis.ipynb
```

### 可視化を確認
```bash
# 画像ビューアで開く
eog weight_distributions.png
eog weight_l2_norm_comparison.png
eog weights_heatmap.png
eog weight_effect_correlation.png
```

---

## データソース

- **検証結果**: `../../results/base_vs_steering/`
- **最適化済み重み**: `../../optimization_results_26personas/`
- **元データ**: 2026-01-02〜01-04の26 personas最適化実験

---

## 関連実験

- **4条件比較**: `../../results/four_conditions/`
- **Fitness比較**: `../../results/fitness_comparison/`
- **Data quality selection**: `../../results/data_selection/`

---

## 引用

この分析結果を論文で引用する場合:

```
Base vs Steering Analysis (2026). Master Thesis Experiment.
Nakata Lab, 2026-01-07.
Finding: Activation steering shows effect only for personas
with L2 norm > 5.0 (4/26 personas, 15.4%).
```

---

## 連絡先

質問・コメント: 中田研究室

**分析実施**: Claude Code (Anthropic)
**実験設計**: 中田研究室
