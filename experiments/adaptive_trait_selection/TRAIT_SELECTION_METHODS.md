# Adaptive Trait Selection 方法論

**目的**: 各ペルソナに最も効果的なTrait組み合わせを自動的に決定する

---

## 1. アプローチ比較

### Method 1: Exhaustive Search（全探索）

**概要**: 全32通り（2^5）のTrait組み合わせを試す

**手順**:
```
For each persona:
    For each trait combination (R1, R2, R3, R4, R5 の部分集合):
        - 短期最適化を実行（5-10世代）
        - Fitnessを記録
    Best combinationを選択
    Full optimization（20-50世代）を実行
```

**計算コスト**:
- 1ペルソナあたり: 32回の短期最適化 + 1回のフル最適化
- 時間: 約32 × 30分 = 16時間/ペルソナ

**メリット**:
- ✅ 確実に最適な組み合わせを発見
- ✅ Trait間の相互作用も捉えられる

**デメリット**:
- ❌ 計算コストが非常に高い
- ❌ 26ペルソナだと416時間（17日）必要

**推奨度**: ⭐⭐☆☆☆ （計算コストが高すぎる）

---

### Method 2: Sequential Forward Selection（逐次前向き選択）

**概要**: 空集合から始めて、1つずつTraitを追加していく

**手順**:
```python
selected_traits = []
remaining_traits = [R1, R2, R3, R4, R5]

while remaining_traits:
    best_trait = None
    best_fitness = current_fitness

    # 各候補Traitを試す
    for trait in remaining_traits:
        temp_traits = selected_traits + [trait]
        fitness = evaluate_trait_combination(temp_traits, generations=5)

        if fitness > best_fitness:
            best_fitness = fitness
            best_trait = trait

    # 改善がなければ終了
    if best_trait is None:
        break

    selected_traits.append(best_trait)
    remaining_traits.remove(best_trait)

# 最終最適化
final_weights = optimize_full(selected_traits, generations=50)
```

**計算コスト**:
- 最悪ケース: 1 + 2 + 3 + 4 + 5 = 15回の短期最適化
- 平均ケース: ~10回（通常2-3 Traitsで収束）
- 時間: 約10 × 30分 = 5時間/ペルソナ

**メリット**:
- ✅ 計算コストが現実的（26ペルソナで130時間 ≈ 5.4日）
- ✅ 解釈可能性が高い（どのTraitが重要か明確）
- ✅ 早期終了が可能

**デメリット**:
- ⚠️ 局所最適解に陥る可能性（TraitのRedundancyを考慮できない）

**推奨度**: ⭐⭐⭐⭐☆ （バランスが良い）

---

### Method 3: Sequential Backward Elimination（逐次後向き除去）

**概要**: 全Traitsから始めて、1つずつ除去していく

**手順**:
```python
selected_traits = [R1, R2, R3, R4, R5]
current_fitness = evaluate_trait_combination(selected_traits, generations=5)

while len(selected_traits) > 1:
    worst_trait = None
    best_fitness_after_removal = 0

    # 各Traitを除去して効果を確認
    for trait in selected_traits:
        temp_traits = [t for t in selected_traits if t != trait]
        fitness = evaluate_trait_combination(temp_traits, generations=5)

        # 除去してもパフォーマンスが落ちない（または改善）
        if fitness >= current_fitness:
            if fitness > best_fitness_after_removal:
                best_fitness_after_removal = fitness
                worst_trait = trait

    # 除去できるTraitがなければ終了
    if worst_trait is None:
        break

    selected_traits.remove(worst_trait)
    current_fitness = best_fitness_after_removal

# 最終最適化
final_weights = optimize_full(selected_traits, generations=50)
```

**計算コスト**:
- 最悪ケース: 5 + 4 + 3 + 2 = 14回の短期最適化
- 平均ケース: ~8回
- 時間: 約8 × 30分 = 4時間/ペルソナ

**メリット**:
- ✅ Forward Selectionより若干速い
- ✅ RedundantなTraitを効率的に除去

**デメリット**:
- ⚠️ 初期評価が5-Trait全てで必要（重い）
- ⚠️ 相補的なTraitを誤って除去する可能性

**推奨度**: ⭐⭐⭐☆☆

---

### Method 4: LASSO-like Trait Selection（統計的アプローチ）

**概要**: 各Traitの単独効果と相互作用を統計的に分析

**手順**:
```python
# Phase 1: 単独効果の測定
for trait in [R1, R2, R3, R4, R5]:
    fitness = evaluate_single_trait(trait, generations=5)
    trait_importance[trait] = fitness

# Phase 2: ペアワイズ相互作用
top_traits = select_top_k(trait_importance, k=3)
for trait1, trait2 in combinations(top_traits, 2):
    fitness = evaluate_trait_combination([trait1, trait2], generations=5)
    interaction_score[(trait1, trait2)] = fitness

# Phase 3: 最適組み合わせの決定
selected_traits = optimize_combination_based_on_scores()

# Phase 4: 最終最適化
final_weights = optimize_full(selected_traits, generations=50)
```

**計算コスト**:
- 単独評価: 5回
- ペアワイズ: C(3,2) = 3回
- 合計: 約8回の短期最適化
- 時間: 約4時間/ペルソナ

**メリット**:
- ✅ 計算コスト効率が良い
- ✅ Trait間の相互作用を捉えられる
- ✅ 解釈可能性が非常に高い

**デメリット**:
- ⚠️ 3次以上の相互作用は見逃す可能性

**推奨度**: ⭐⭐⭐⭐⭐ （最も効率的で解釈可能）

---

### Method 5: Hybrid Approach（推奨）

**概要**: Method 4（統計的分析） + Method 2（Forward Selection）の組み合わせ

**手順**:

```python
# PHASE 1: Quick Statistical Screening (約2時間)
# ─────────────────────────────────────────
# 各Traitの単独効果を測定
trait_scores = {}
for trait in [R1, R2, R3, R4, R5]:
    weights = optimize_single_trait(trait, generations=5)
    fitness = evaluate_fitness(weights)
    trait_scores[trait] = {
        'fitness': fitness,
        'l2_norm': np.linalg.norm(weights)
    }

# 効果が低いTraitを除外（fitness < threshold）
threshold = median(trait_scores.values())
candidate_traits = [t for t in traits if trait_scores[t]['fitness'] > threshold]

print(f"候補Traits: {candidate_traits}")  # 通常2-4個に絞られる


# PHASE 2: Pairwise Interaction Analysis (約1時間)
# ─────────────────────────────────────────
interaction_matrix = {}
for trait1, trait2 in combinations(candidate_traits, 2):
    weights = optimize_trait_pair(trait1, trait2, generations=5)
    fitness = evaluate_fitness(weights)
    interaction_matrix[(trait1, trait2)] = fitness

# 相互作用が強いペアを特定
best_pair = max(interaction_matrix, key=interaction_matrix.get)


# PHASE 3: Incremental Selection (約2時間)
# ─────────────────────────────────────────
selected_traits = list(best_pair)
remaining_traits = [t for t in candidate_traits if t not in selected_traits]

current_fitness = interaction_matrix[best_pair]

while remaining_traits:
    best_addition = None
    best_new_fitness = current_fitness

    for trait in remaining_traits:
        temp_traits = selected_traits + [trait]
        weights = optimize_traits(temp_traits, generations=5)
        fitness = evaluate_fitness(weights)

        if fitness > best_new_fitness:
            best_new_fitness = fitness
            best_addition = trait

    # 改善がなければ終了
    if best_addition is None:
        break

    selected_traits.append(best_addition)
    remaining_traits.remove(best_addition)
    current_fitness = best_new_fitness


# PHASE 4: Full Optimization (約2時間)
# ─────────────────────────────────────────
final_weights = optimize_traits(selected_traits, generations=50)
final_fitness = evaluate_fitness(final_weights)

print(f"Selected Traits: {selected_traits}")
print(f"Final L2 Norm: {np.linalg.norm(final_weights):.2f}")
print(f"Final Fitness: {final_fitness:.4f}")
```

**計算コスト**:
- Phase 1: 5回の短期最適化（30分 × 5 = 2.5時間）
- Phase 2: 3-6回の短期最適化（30分 × 4 = 2時間）
- Phase 3: 1-3回の短期最適化（30分 × 2 = 1時間）
- Phase 4: 1回のフル最適化（2時間）
- **合計: 約7.5時間/ペルソナ**

**メリット**:
- ✅ 効率的（Exhaustiveの半分以下）
- ✅ 堅牢（統計的裏付けあり）
- ✅ 解釈可能（各フェーズの意味が明確）
- ✅ 早期終了可能

**デメリット**:
- ⚠️ 実装が複雑

**推奨度**: ⭐⭐⭐⭐⭐ （最も実用的）

---

## 2. 実装プラン

### Step 1: プロトタイプ実装（1-2ペルソナでテスト）

```bash
# テスト用スクリプト
python scripts/adaptive_trait_selection.py \
    --persona_id episode-225888_A \
    --method hybrid \
    --output_dir experiments/adaptive_trait_selection/
```

### Step 2: バリデーション

既知の結果と比較：
- episode-225888_A: R3, R5が重要と予想される
- episode-184019_A: R2, R4が重要と予想される

### Step 3: 全ペルソナ適用

並列実行で効率化：
- GPU 0: 13ペルソナ
- GPU 1: 13ペルソナ
- 予想時間: 7.5時間（並列実行）

---

## 3. 評価指標

各フェーズで記録：
1. **Fitness**: Style Similarity
2. **L2 Norm**: Steeringの強度
3. **Selected Traits**: どのTraitが選ばれたか
4. **Phase時間**: 各フェーズの実行時間

最終評価：
- Base vs Steering 比較
- 5-Trait vs Adaptive比較
- Tie率の削減

---

## 4. 期待される結果

### 成功基準

✅ **Tier 1 Success**:
- 80%以上のペルソナでL2ノルム > 5.0
- 平均Tie率 < 60%（従来90%）

✅ **Tier 2 Success**:
- 60%以上のペルソナでL2ノルム > 6.0
- 平均Tie率 < 50%

✅ **Tier 3 Success**:
- 全ペルソナでL2ノルム > 4.0
- 平均Tie率 < 70%

### ペルソナ別の予想

| Persona | 予想されるTrait組 | 根拠 |
|---------|-----------------|------|
| episode-225888_A | R3, R4, R5 | 5-Traitで大きな重み |
| episode-184019_A | R2, R4 | 統計的に有意 |
| episode-118328_B | R2, R4 | 統計的に有意 |
| episode-239427_A | R4 single? | 5-TraitでR4が支配的 |

---

## 5. タイムライン（26ペルソナ全体）

| Phase | タスク | 時間 |
|-------|--------|------|
| 1 | プロトタイプ実装 | 2-3時間 |
| 2 | テスト実行（2ペルソナ） | 15時間 |
| 3 | バグ修正・改善 | 2-3時間 |
| 4 | 全ペルソナ実行（並列） | 7.5時間 |
| 5 | 結果分析 | 2-3時間 |
| **合計** | | **約30時間** |

実行開始から完了まで: **約1.5日（並列実行）**

---

## 6. リスクと対策

### Risk 1: 計算時間超過
**対策**: GPU数を増やす（4台使用で4時間に短縮）

### Risk 2: Phase 1で全Trait除外
**対策**: Threshold調整、最低2 Traitsは保持

### Risk 3: 不安定な収束
**対策**: 各フェーズで3回試行し、平均を取る

---

## 7. 次のステップ

実装準備が完了したら:
1. `scripts/adaptive_trait_selection.py` を作成
2. テストペルソナで動作確認
3. 全ペルソナで実行
4. 結果をnotebookで分析
