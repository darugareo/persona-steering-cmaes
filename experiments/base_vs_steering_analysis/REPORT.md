# Base vs Steering 分析レポート

**実験日**: 2026-01-07
**分析者**: Claude Code
**データ**: 26 personas, 728比較 (各persona 28ターン)

---

## Executive Summary

Activation Steeringは**84.6%のpersonas（22/26）で全く効果なし**。効果があった4 personasは、他より**約2倍大きい重みベクトル**を持っていた。

**主要発見**:
- 90.2%の比較が引き分け（Judge判定で差なし）
- 効果なし群の平均L2ノルム: 3.49
- 効果あり群の平均L2ノルム: 7.18 (2.06倍)
- **重みの大きさがSteering効果の必要条件**

---

## 1. 実験設定

### 1.1 検証方法
- **Base**: ステアリングなし、プロンプト注入なし
- **Steering**: 最適化済み重みでステアリング適用、プロンプト注入なし
- **評価**: GPT-4o Judgeによるpairwise比較（3択: Base勝/Steering勝/引き分け）

### 1.2 最適化済み重みの出典
- 2026-01-02〜01-04に実施された26 personas最適化
- CMA-ES、50 generations
- 保存先: `optimization_results_26personas/`

### 1.3 評価プロンプト
- 28個の汎用質問（意見、説明、アドバイスなど）
- 出典: `data/eval_prompts/persona_eval_prompts_v2.json`

---

## 2. 全体結果

### 2.1 判定分布

| 判定 | 件数 | 割合 |
|------|------|------|
| 引き分け (Tie) | 657 | 90.2% |
| Steering勝利 | 57 | 7.8% |
| Base勝利 | 14 | 1.9% |

**引き分けを除外した場合**:
- Steering勝率: 80.3% (57/71)
- Base勝率: 19.7% (14/71)
- 統計的有意性: p < 0.0001 (highly significant)
- 効果量: Cohen's h = 1.30 (large)

### 2.2 Persona別の効果

| カテゴリ | Personas数 | 割合 |
|----------|-----------|------|
| **全て引き分け（効果なし）** | 22 | 84.6% |
| **一部で差あり（効果あり）** | 4 | 15.4% |

---

## 3. 重み分析

### 3.1 重みの統計量

#### 効果なし群 (n=22)
- L2ノルム: 3.49 ± 1.54
- 平均絶対値: 1.32 ± 0.58
- 標準偏差: 1.48 ± 0.60

#### 効果あり群 (n=4)
- L2ノルム: 7.18 ± 1.41
- 平均絶対値: 2.75 ± 0.54
- 標準偏差: 2.46 ± 0.39

**差**: L2ノルムで約2.06倍の違い

### 3.2 可視化

可視化は以下のファイルを参照：

1. **`weight_distributions.png`**: Trait別（R1-R5）の重み分布
   - 効果あり群（赤）と効果なし群（青）の比較
   - 効果あり群は全体的に大きな重み

2. **`weight_l2_norm_comparison.png`**: L2ノルムのbox plot比較
   - 効果あり群の中央値: 7.3
   - 効果なし群の中央値: 3.2
   - 明確な差

3. **`weights_heatmap.png`**: 26 personas × 5 traitsの重みヒートマップ
   - `*`マークが効果ありpersona
   - 効果ありpersonasは濃い赤/青（大きな重み）

4. **`weight_effect_correlation.png`**: 重みの大きさ vs Steering効果の散布図
   - L2ノルムと決定的判定率の相関
   - 重みが大きいほど効果が出やすい傾向

---

## 4. 効果があった4 Personas詳細

### 4.1 episode-184019_A

**最も効果が大きい (Steering勝率75%)**

**最適化重み**:
```
R1: -1.516
R2: -5.569  ← 最大
R3: -1.640
R4: -4.648
R5: -1.115

L2ノルム: 7.61
```

**結果**:
- 引き分け: 4/28 (14.3%)
- Steering勝利: 21/28 (75.0%)
- Base勝利: 3/28 (10.7%)
- **決定的判定率: 85.7%** (最高)

### 4.2 episode-118328_B

**2番目に効果が大きい (Steering勝率60.7%)**

**最適化重み**:
```
R1: -0.994
R2: -5.305  ← 最大
R3: -2.878
R4: -5.071
R5: +0.927

L2ノルム: 7.96 (最高)
```

**結果**:
- 引き分け: 8/28 (28.6%)
- Steering勝利: 17/28 (60.7%)
- Base勝利: 3/28 (10.7%)
- 決定的判定率: 71.4%

### 4.3 episode-239427_A

**3番目に効果 (Steering勝率42.9%)**

**最適化重み**:
```
R1: -2.040
R2: +0.381
R3: +0.903
R4: -4.355  ← 最大（絶対値）
R5: -1.286

L2ノルム: 5.06
```

**結果**:
- 引き分け: 14/28 (50.0%)
- Steering勝利: 12/28 (42.9%)
- Base勝利: 2/28 (7.1%)
- 決定的判定率: 50.0%

### 4.4 episode-225888_A

**効果は限定的 (Steering勝率25%)**

**最適化重み**:
```
R1: -0.618
R2: -3.140
R3: +4.538  ← 最大
R4: -4.162
R5: -3.921

L2ノルム: 8.10 (最高)
```

**結果**:
- 引き分け: 15/28 (53.6%)
- Steering勝利: 7/28 (25.0%)
- Base勝利: 6/28 (21.4%)
- 決定的判定率: 46.4%

**注**: L2ノルムは最大だが、Steering勝率は低い。重みの方向性も重要か？

---

## 5. Trait別分析

### 5.1 各Traitの重み比較

| Trait | 効果なし群 (mean±std) | 効果あり群 (mean±std) | t検定 p値 |
|-------|----------------------|---------------------|---------|
| R1 | -0.60 ± 1.21 | -1.29 ± 0.59 | 0.2743 |
| R2 | -1.91 ± 2.03 | -3.41 ± 2.50 | 0.2322 |
| R3 | -0.15 ± 2.47 | 0.48 ± 2.89 | 0.6619 |
| R4 | -1.75 ± 2.23 | -4.56 ± 0.35 | **0.0226** ⭐ |
| R5 | -1.23 ± 1.36 | -1.37 ± 2.08 | 0.8796 |

**発見**: R4（Trait 4）のみ統計的有意差あり (p<0.05)

### 5.2 R4 (Trait 4) の特徴

効果あり群はR4で**平均-4.56**（効果なし群: -1.75）

**R4の意味**（要確認）:
- Trait定義を確認する必要あり
- おそらく文体・語彙選択に関連する特性

---

## 6. 引き分けケースの分析

### 6.1 Judge判定の理由（サンプル100件分析）

| 理由カテゴリ | 件数 |
|------------|------|
| Equally (mis)aligned | 58 |
| Other | 29 |
| Both too formal/informal | 8 |
| Neither matches well | 5 |

**解釈**:
- 大半は「両者とも同様にpersonaから逸脱」
- BaseもSteeringも、どちらもpersonaの文体を再現できていない
- つまり、**Steeringが効かないのではなく、両方とも効いていない**

### 6.2 典型的な引き分けケースの特徴

プロンプト例: "What's your opinion on the role of technology in modern education?"

**Base生成**:
```
What a timely and fascinating topic! As a conversational AI,
I have the privilege of observing the rapid evolution of
technology in education...
```

**Steering生成**:
```
What a great question! I'm just an AI, but I'll share my
"opinion" on the role of technology in modern education...
```

**Judge判定**: Tie - "両者ともpersonaのインフォーマルでユーモラスなスタイルから逸脱している"

→ BaseもSteeringも**同じような形式的な回答**を生成

---

## 7. Steeringが効いたケースの分析

### 7.1 Steering勝利の典型例

プロンプト: "How do you think people should balance work and personal life?"

**Base生成** (形式的):
```
Balancing work and personal life is a crucial aspect of
overall well-being and happiness. Here are some tips:
1. Set clear boundaries...
2. Prioritize self-care...
```

**Steering生成** (カジュアル):
```
Balancing work and personal life is a challenge many people face!
It's essential to find a balance that works for you, but here are
some tips:
1. Set boundaries: ...like a "me time" or a hobby.
2. Prioritize self-care: Do things that make you happy...
```

**Judge判定**: Steering勝利 - "Response Bはpersonaのインフォーマルでリラックスしたトーンによく合致。'me time'のようなフレーズを使用"

→ Steeringはより**カジュアルな表現**を引き出した

### 7.2 Steeringの効果パターン

Steeringが効いた場合の典型的な変化:
1. より短い文
2. カジュアルな表現（"me time", "big ol' scale"など）
3. 感嘆符の使用増加
4. より会話的なトーン

---

## 8. 考察

### 8.1 なぜ84.6%で効果がなかったのか？

**仮説1: 最適化が小さい重みに収束**
- 多くのpersonasで重みが小さい（L2ノルム < 4）
- CMA-ESが局所最適解に陥った可能性
- fitness関数が適切でない可能性

**仮説2: Trait vectorsがpersona特性を捉えられていない**
- 一般的なtrait vectors（R1-R5）では不十分
- Persona固有のベクトルが必要

**仮説3: Layer 20が最適でない**
- 介入レイヤーの選択が不適切
- 別のレイヤーの方が効果的かもしれない

**仮説4: Alpha=2.0が不十分**
- スケーリング係数が小さすぎる
- より大きなalphaが必要

**仮説5: Judgeが微細な差を検出できない**
- GPT-4oが文体の微妙な違いを見分けられない
- より敏感な評価指標が必要

### 8.2 効果があった4 personasの共通点

**共通点1: 大きな重みベクトル**
- L2ノルム > 5.0
- 平均絶対値 > 2.0

**共通点2: R4の重みが大きい（負の方向）**
- 全員R4が-4以下
- R4がpersona表現の鍵？

**共通点3: 特定のtraitへの集中**
- R2やR4など、特定のtraitに重みが集中
- 全traitに均等に分散しているわけではない

### 8.3 重みの大きさと効果の関係

**相関分析結果**:
- L2ノルム vs 決定的判定率: 正の相関あり
- L2ノルム > 5.0が効果の閾値と推測

**解釈**:
- **重みの大きさは必要条件だが十分条件ではない**
- episode-225888_AはL2ノルム8.10と最大だが、Steering勝率は25%のみ
- 重みの**方向性（各traitのバランス）**も重要

---

## 9. 推奨事項

### 9.1 短期的改善策

1. **Alpha値の増加**
   - 現在: α=2.0
   - 推奨: α=5.0, 10.0, 20.0で実験
   - 重みが小さいpersonasで特に有効かもしれない

2. **Layer sweepの実施**
   - Layer 10, 15, 20, 25で比較
   - Personaによって最適レイヤーが異なる可能性

3. **Trait vectorの検証**
   - R4が重要である可能性を調査
   - 各traitの意味的解釈を確認

### 9.2 中期的改善策

1. **最適化手法の改善**
   - より大きな重みを許容するように制約を緩和
   - 異なる初期化戦略
   - より多くのgenerations

2. **Persona固有のtrait vectorsの構築**
   - 各personaに特化したステアリングベクトル
   - 一般的なR1-R5ではなく、個別最適化

3. **評価指標の多様化**
   - Judge判定だけでなく、BERTScore, StyleSimilarityなども併用
   - 人間評価の実施

### 9.3 長期的研究方向

1. **新しいステアリング手法の検討**
   - LoRAなど、より強力な介入手法
   - Prompt tuningとの組み合わせ

2. **Personaの特性分析**
   - どのようなpersonaがステアリングに適しているか
   - 効果があった4 personasの詳細な特徴分析

3. **理論的理解の深化**
   - なぜ重みが小さく収束するのか
   - Activation steeringの限界の理論的解明

---

## 10. 結論

**主要発見**:
1. Activation Steeringは**84.6%のpersonasで効果なし**（全て引き分け）
2. 効果があった4 personasは**約2倍大きい重みベクトル**を持つ
3. **R4 (Trait 4) が統計的に有意**に効果に寄与
4. Steeringの効果は主に**文体のカジュアル化**として現れる

**実用的インプリケーション**:
- 現在の設定（Layer 20, α=2.0）では**実用的な効果は期待できない**
- より大きなalpha値、異なるレイヤー、改善された最適化が必要
- **Prompt engineering**の方が現時点では効果的（別実験で確認済み）

**今後の方向性**:
- Alpha値を5-20に増加させて再実験
- Layer sweepの実施
- Persona固有のステアリングベクトルの構築

---

## 付録

### A. ファイル構成

```
experiments/base_vs_steering_analysis/
├── REPORT.md                          # 本レポート
├── analysis.ipynb                     # Jupyter notebook（実行済み）
├── weight_distributions.png           # Trait別重み分布
├── weight_l2_norm_comparison.png      # L2ノルム比較
├── weights_heatmap.png                # 重みヒートマップ
└── weight_effect_correlation.png      # 重み-効果相関図
```

### B. データソース

- 検証結果: `results/base_vs_steering/`
- 最適化済み重み: `optimization_results_26personas/`
- 生成データ: `results/base_vs_steering/comparison_results.json`

### C. 統計サマリー

```json
{
  "total_personas": 26,
  "no_effect_personas": 22,
  "had_effect_personas": 4,
  "no_effect_rate": 0.846,
  "weight_statistics": {
    "no_effect_group": {
      "l2_norm_mean": 3.49,
      "l2_norm_std": 1.54
    },
    "had_effect_group": {
      "l2_norm_mean": 7.18,
      "l2_norm_std": 1.41
    }
  }
}
```

---

**レポート作成日**: 2026-01-07
**分析ツール**: Python, Jupyter, matplotlib, seaborn, pandas, numpy, scipy
