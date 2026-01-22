# 2-Trait Steering実験計画書

**実験開始**: 2026-01-07
**予想完了**: 2026-01-07 (1-2時間後)

---

## 1. 背景

### 1.1 問題
- 5-Trait Steering（R1-R5）は**84.6%のpersonasで効果なし**
- 重みが小さい（L2ノルム < 4）ため、Steeringが効かない

### 1.2 原因分析
Trait別の効果分析により判明：
- **R4 (Trait 4)**: p < 0.001, 差=+3.049 → **highly significant**
- **R2 (Trait 2)**: p = 0.0012, 差=+2.344 → **significant**
- **R3 (Trait 3)**: p = 0.1944, 差=+1.082 → marginal
- **R5 (Trait 5)**: p = 0.4427, 差=+0.480 → **no effect**
- **R1 (Trait 1)**: p = 0.6156, 差=+0.177 → **no effect**

→ **R1とR5は最適化を邪魔している**可能性

### 1.3 仮説
**不要なTrait（R1, R3, R5）を除去し、R2とR4のみで最適化すれば、より大きな重みに収束し、Steering効果が向上する**

---

## 2. 実験設計

### 2.1 変更点
| 項目 | 5-Trait (従来) | 2-Trait (新) |
|------|---------------|------------|
| 使用Traits | R1, R2, R3, R4, R5 | **R2, R4のみ** |
| 最適化次元 | 5次元 | **2次元** |
| 探索空間 | 5D空間 | **2D平面** |
| パラメータ数 | 5 | **2** |

### 2.2 期待される効果
1. **探索効率の向上**: 5次元 → 2次元で探索が容易に
2. **収束の改善**: 不要な次元がないため、より大きな重みに収束
3. **L2ノルムの増加**: 5-Traitの ~3.5 → 2-Traitで > 5.0を期待
4. **Steering効果の向上**: Tie率 90% → < 70%を目標

### 2.3 実験対象
効果があった4 personas（5-Trait時代）:
1. **episode-184019_A**: 5-Trait時 Steering勝率 75%, L2ノルム 7.61
2. **episode-118328_B**: 5-Trait時 Steering勝率 60.7%, L2ノルム 7.96
3. **episode-239427_A**: 5-Trait時 Steering勝率 42.9%, L2ノルム 5.06
4. **episode-225888_A**: 5-Trait時 Steering勝率 25%, L2ノルム 8.10

---

## 3. 実験パラメータ

### 3.1 最適化設定
```python
dimensions = 2  # R2, R4
initial_mean = [0.0, 0.0]
initial_sigma = 2.0  # 従来通り
population_size = 8
max_generations = 20
fitness_function = "Style Similarity"  # 最も汎化性能が高かった
```

### 3.2 Steering設定
```python
layer = 20  # 従来通り
alpha = 2.0  # 従来通り（後で調整可能）
model = "meta-llama/Meta-Llama-3-8B-Instruct"
```

### 3.3 評価設定
```python
train_turns = 10  # 最適化用
test_turns = 10   # 評価用（selected）
```

---

## 4. 実行状況

### 4.1 最適化（Phase 1）- 進行中

**試行1 (失敗)**: 2026-01-07 01:57
- エラー: `ModuleNotFoundError: No module named 'persona_opt.fitness_functions'`
- 対応: style similarity計算を直接実装

**試行2 (現在実行中)**: 2026-01-07 02:21

**GPU割り当て**:
- GPU 0: episode-184019_A, episode-239427_A
- GPU 1: episode-118328_B, episode-225888_A

**プロセスID**:
- 1922449: episode-184019_A (CPU: 105%, MEM: 1.47GB)
- 1922697: episode-239427_A (CPU: 103%, MEM: 1.49GB)
- 1922698: episode-118328_B (CPU: 102%, MEM: 1.50GB)
- 1923068: episode-225888_A (CPU: 105%, MEM: 1.47GB)

**ログファイル**:
```
logs/two_trait_optimization/episode-184019_A.log
logs/two_trait_optimization/episode-118328_B.log
logs/two_trait_optimization/episode-239427_A.log
logs/two_trait_optimization/episode-225888_A.log
```

**ステータス**: モデル読み込み完了、最適化開始準備中
**予想完了**: 2026-01-07 04:00-05:00 (1.5-2.5時間)

### 4.2 評価（Phase 2）- 未開始

最適化完了後に実行:
1. 2-Trait Steeringでgeneration
2. Base vs 2-Trait Steering比較
3. 5-Trait vs 2-Trait比較

---

## 5. 評価指標

### 5.1 最適化の成功判定
✅ L2ノルムが5-Traitより大きい（> 5.0目標）
✅ 収束世代数が少ない（< 15世代）
✅ 最終fitnessが5-Traitより高い

### 5.2 Steering効果の成功判定
✅ Tie率が90%から減少（< 70%目標）
✅ Steering勝率が5-Traitより高い
✅ Judge信頼度が向上（> 3.5）

### 5.3 比較表（目標値）

| 指標 | 5-Trait | 2-Trait（目標） |
|------|---------|----------------|
| L2ノルム平均 | 7.18 | **> 8.0** |
| Steering勝率（4 personas平均） | 50.9% | **> 60%** |
| Tie率 | 36.6% | **< 30%** |
| 最適化世代数 | ~50 | **< 20** |

---

## 6. 次のステップ（Phase 3以降）

### Phase 3: 成功した場合
1. ✅ 残り22 personasでも2-Trait最適化を実施
2. ✅ Alpha値の調整実験（α=2.0 → 5.0, 10.0）
3. ✅ Layer sweepの実施（Layer 15, 20, 25）

### Phase 4: 失敗した場合
1. ❌ R4単体での最適化（1-Trait）
2. ❌ R2+R4+R3での最適化（3-Trait）
3. ❌ 異なるlayerでの実験

---

## 7. データ保存先

### 7.1 最適化結果
```
optimization_results_2trait/
├── episode-184019_A/
│   ├── episode-184019_A_weights.json  # {"R2": X, "R4": Y}
│   └── episode-184019_A_log.json      # 最適化ログ
├── episode-118328_B/
├── episode-239427_A/
└── episode-225888_A/
```

### 7.2 評価結果（Phase 2後）
```
results/two_trait_evaluation/
├── generations/              # 生成結果
├── comparisons/              # Base vs Steering
├── 5trait_vs_2trait.json    # 比較結果
└── SUMMARY.md               # サマリーレポート
```

### 7.3 実験パッケージ（Phase 3後）
```
experiments/two_trait_steering/
├── EXPERIMENT_PLAN.md       # 本ファイル
├── RESULTS.md               # 結果レポート
├── analysis.ipynb           # 分析notebook
└── figures/                 # 可視化
    ├── l2_norm_comparison.png
    ├── win_rate_comparison.png
    └── convergence_plot.png
```

---

## 8. リスクと対策

### Risk 1: 2-Traitでも重みが小さい
**対策**: Initial sigmaを増やす（2.0 → 4.0）

### Risk 2: 収束が遅い
**対策**: Population sizeを増やす（8 → 16）

### Risk 3: 局所最適解に陥る
**対策**: 複数の初期値から実験（[0,0], [2,-2], [-2,2]など）

---

## 9. タイムライン

| 日時 | タスク | 状態 |
|------|--------|------|
| 2026-01-07 01:57 | Phase 1: 最適化開始 (試行1 - 失敗) | ❌ ModuleNotFoundError |
| 2026-01-07 02:21 | Phase 1: 最適化再開 (試行2) | ✅ 実行中 |
| 2026-01-07 04:00 | Phase 1: 最適化完了予定 | ⏳ 進行中 |
| 2026-01-07 05:00 | Phase 2: 評価開始 | ⏸️ 待機中 |
| 2026-01-07 06:00 | Phase 2: 評価完了予定 | ⏸️ 待機中 |
| 2026-01-07 07:00 | 結果分析とレポート作成 | ⏸️ 待機中 |

---

## 10. 成功基準（まとめ）

### 最低限の成功 (Minimum Viable Success)
- [ ] 少なくとも1 personaでL2ノルム > 5.0
- [ ] 少なくとも1 personaでTie率 < 50%

### 部分的成功 (Partial Success)
- [ ] 4 personas中2つでL2ノルム > 5.0
- [ ] 平均Tie率 < 60%
- [ ] 平均Steering勝率 > 5-Trait

### 完全な成功 (Full Success)
- [ ] 全4 personasでL2ノルム > 5.0
- [ ] 平均Tie率 < 40%
- [ ] 平均Steering勝率 > 70%
- [ ] 収束世代数 < 15

---

**実験責任者**: Claude Code (Anthropic)
**データ管理**: 中田研究室
**予算**: GPU時間 4-8時間
