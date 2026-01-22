#!/usr/bin/env python3
"""
Adaptive Trait Selection: ペルソナごとに最適なTrait組み合わせを選択

各ペルソナの5-Trait最適化結果から、重要度の高いTraitを特定し、
推奨Trait組み合わせを出力する。
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


class PersonaTraitSelector:
    """ペルソナごとに最適なTrait組み合わせを選択"""

    def __init__(self, results_dir: str = "optimization_results_26personas"):
        self.results_dir = Path(results_dir)
        self.personas_data = self._load_all_results()

    def _load_all_results(self) -> Dict:
        """全ペルソナの5-Trait最適化結果を読み込む"""
        personas = {}

        for result_file in self.results_dir.glob("gpu*/*_optimization.json"):
            with open(result_file) as f:
                data = json.load(f)
                persona_id = data["persona_id"]
                personas[persona_id] = {
                    "weights": data["best_weights"],
                    "score": data["best_score"],
                    "l2_norm": np.sqrt(sum(w**2 for w in data["best_weights"].values()))
                }

        print(f"✓ Loaded {len(personas)} persona results")
        return personas

    def analyze_trait_importance(
        self,
        persona_id: str,
        method: str = "absolute"
    ) -> List[Tuple[str, float]]:
        """
        ペルソナのTrait重要度を分析

        Args:
            persona_id: ペルソナID
            method: 重要度計算方法
                - "absolute": 絶対値
                - "squared": 二乗（L2への寄与度）
                - "percentage": L2ノルムに対する割合

        Returns:
            [(trait_name, importance_score), ...] を重要度降順でソート
        """
        if persona_id not in self.personas_data:
            raise ValueError(f"Persona {persona_id} not found")

        weights = self.personas_data[persona_id]["weights"]
        l2_norm = self.personas_data[persona_id]["l2_norm"]

        importance = []

        for trait, weight in weights.items():
            if method == "absolute":
                score = abs(weight)
            elif method == "squared":
                score = weight ** 2
            elif method == "percentage":
                score = (weight ** 2) / (l2_norm ** 2) * 100
            else:
                raise ValueError(f"Unknown method: {method}")

            importance.append((trait, score))

        # 重要度降順でソート
        importance.sort(key=lambda x: x[1], reverse=True)

        return importance

    def select_traits(
        self,
        persona_id: str,
        selection_method: str = "top_k",
        k: int = 2,
        threshold: float = 1.0
    ) -> List[str]:
        """
        ペルソナに最適なTrait組み合わせを選択

        Args:
            persona_id: ペルソナID
            selection_method:
                - "top_k": 上位k個のTraitを選択
                - "threshold": 絶対重み > threshold のTraitを選択
                - "cumulative": L2への累積寄与率が指定%に達するまで選択
            k: top_k法で選択する数
            threshold: threshold法での閾値

        Returns:
            選択されたTrait名のリスト
        """
        importance = self.analyze_trait_importance(persona_id, method="absolute")

        if selection_method == "top_k":
            selected = [trait for trait, _ in importance[:k]]

        elif selection_method == "threshold":
            selected = [trait for trait, score in importance if score >= threshold]

        elif selection_method == "cumulative":
            # L2への寄与率で選択
            importance_pct = self.analyze_trait_importance(persona_id, method="percentage")
            cumulative = 0.0
            selected = []
            target_pct = threshold  # この場合thresholdは累積%（例: 80.0）

            for trait, pct in importance_pct:
                selected.append(trait)
                cumulative += pct
                if cumulative >= target_pct:
                    break

        else:
            raise ValueError(f"Unknown selection method: {selection_method}")

        return selected

    def generate_recommendations(
        self,
        selection_method: str = "top_k",
        k: int = 2,
        threshold: float = 1.5,
        output_file: str = "trait_recommendations.json"
    ):
        """
        全ペルソナの推奨Trait組み合わせを生成

        Args:
            selection_method: 選択方法
            k: top_kの場合の選択数
            threshold: thresholdまたはcumulative法の閾値
            output_file: 出力ファイル名
        """
        recommendations = {}
        trait_usage_count = {"R1": 0, "R2": 0, "R3": 0, "R4": 0, "R5": 0}

        print(f"\n{'='*80}")
        print(f"Trait Selection: method={selection_method}, k={k}, threshold={threshold}")
        print(f"{'='*80}\n")

        for persona_id in sorted(self.personas_data.keys()):
            # Trait重要度分析
            importance = self.analyze_trait_importance(persona_id, method="absolute")

            # Trait選択
            selected_traits = self.select_traits(
                persona_id,
                selection_method=selection_method,
                k=k,
                threshold=threshold
            )

            # 選択されたTraitの重みと寄与率
            weights = self.personas_data[persona_id]["weights"]
            l2_norm = self.personas_data[persona_id]["l2_norm"]

            selected_weights = {trait: weights[trait] for trait in selected_traits}
            selected_l2 = np.sqrt(sum(w**2 for w in selected_weights.values()))
            retention_ratio = (selected_l2 / l2_norm) * 100

            # カウント
            for trait in selected_traits:
                trait_usage_count[trait] += 1

            # 推奨情報を記録
            recommendations[persona_id] = {
                "selected_traits": selected_traits,
                "selected_weights": selected_weights,
                "all_weights": weights,
                "trait_importance": [
                    {"trait": trait, "abs_weight": float(score)}
                    for trait, score in importance
                ],
                "l2_norm_original": float(l2_norm),
                "l2_norm_selected": float(selected_l2),
                "retention_ratio_pct": float(retention_ratio),
                "score": float(self.personas_data[persona_id]["score"])
            }

            # 出力
            print(f"{persona_id}:")
            print(f"  Selected: {', '.join(selected_traits)}")
            print(f"  Weights: {', '.join(f'{t}={selected_weights[t]:+.2f}' for t in selected_traits)}")
            print(f"  L2: {l2_norm:.2f} → {selected_l2:.2f} ({retention_ratio:.1f}% retained)")
            print()

        # サマリー統計
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        print(f"Total personas: {len(recommendations)}")
        print(f"\nTrait usage frequency:")
        for trait in ["R1", "R2", "R3", "R4", "R5"]:
            count = trait_usage_count[trait]
            pct = count / len(recommendations) * 100
            print(f"  {trait}: {count:2d}/{len(recommendations)} ({pct:5.1f}%)")

        # L2ノルム保持率の統計
        retention_ratios = [rec["retention_ratio_pct"] for rec in recommendations.values()]
        print(f"\nL2 norm retention:")
        print(f"  Mean: {np.mean(retention_ratios):.1f}%")
        print(f"  Std:  {np.std(retention_ratios):.1f}%")
        print(f"  Min:  {np.min(retention_ratios):.1f}%")
        print(f"  Max:  {np.max(retention_ratios):.1f}%")
        print(f"{'='*80}\n")

        # 結果を保存
        output = {
            "selection_config": {
                "method": selection_method,
                "k": k,
                "threshold": threshold
            },
            "summary": {
                "total_personas": len(recommendations),
                "trait_usage": trait_usage_count,
                "retention_stats": {
                    "mean_pct": float(np.mean(retention_ratios)),
                    "std_pct": float(np.std(retention_ratios)),
                    "min_pct": float(np.min(retention_ratios)),
                    "max_pct": float(np.max(retention_ratios))
                }
            },
            "recommendations": recommendations
        }

        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"✅ Recommendations saved to: {output_path}")

        return recommendations

    def compare_selection_methods(self):
        """複数の選択方法を比較"""
        methods = [
            {"name": "Top-2", "method": "top_k", "k": 2},
            {"name": "Top-3", "method": "top_k", "k": 3},
            {"name": "Threshold-1.0", "method": "threshold", "threshold": 1.0},
            {"name": "Threshold-1.5", "method": "threshold", "threshold": 1.5},
            {"name": "Threshold-2.0", "method": "threshold", "threshold": 2.0},
            {"name": "Cumulative-80%", "method": "cumulative", "threshold": 80.0},
            {"name": "Cumulative-90%", "method": "cumulative", "threshold": 90.0}
        ]

        comparison = []

        for config in methods:
            name = config.pop("name")
            selection_method = config.pop("method")

            # 一時的に推奨を生成（出力なし）
            trait_counts = {"R1": 0, "R2": 0, "R3": 0, "R4": 0, "R5": 0}
            retention_ratios = []
            trait_count_per_persona = []

            for persona_id in self.personas_data.keys():
                selected = self.select_traits(persona_id, selection_method=selection_method, **config)

                for trait in selected:
                    trait_counts[trait] += 1

                trait_count_per_persona.append(len(selected))

                # Retention ratio
                weights = self.personas_data[persona_id]["weights"]
                l2_orig = self.personas_data[persona_id]["l2_norm"]
                selected_weights = {trait: weights[trait] for trait in selected}
                l2_selected = np.sqrt(sum(w**2 for w in selected_weights.values()))
                retention_ratios.append((l2_selected / l2_orig) * 100)

            comparison.append({
                "method": name,
                "config": {"selection_method": selection_method, **config},
                "avg_traits_per_persona": float(np.mean(trait_count_per_persona)),
                "avg_retention_pct": float(np.mean(retention_ratios)),
                "trait_usage": trait_counts
            })

        # 比較表を表示
        print(f"\n{'='*80}")
        print("SELECTION METHOD COMPARISON")
        print(f"{'='*80}")
        print(f"{'Method':<20} {'Avg #Traits':>12} {'Avg Retention':>14} {'Trait Usage'}")
        print("-" * 80)

        for comp in comparison:
            usage_str = " ".join(f"{t}:{comp['trait_usage'][t]}" for t in ["R1", "R2", "R3", "R4", "R5"])
            print(f"{comp['method']:<20} {comp['avg_traits_per_persona']:>12.1f} "
                  f"{comp['avg_retention_pct']:>13.1f}% {usage_str}")

        print(f"{'='*80}\n")

        return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Trait Selection: ペルソナごとに最適なTrait組み合わせを選択"
    )
    parser.add_argument(
        "--method",
        choices=["top_k", "threshold", "cumulative"],
        default="threshold",
        help="選択方法"
    )
    parser.add_argument("--k", type=int, default=2, help="top_k法での選択数")
    parser.add_argument("--threshold", type=float, default=1.5, help="threshold法での閾値（または累積%）")
    parser.add_argument(
        "--results_dir",
        default="optimization_results_26personas",
        help="最適化結果のディレクトリ"
    )
    parser.add_argument(
        "--output",
        default="trait_recommendations.json",
        help="出力ファイル名"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="複数の選択方法を比較"
    )

    args = parser.parse_args()

    # Selectorを作成
    selector = PersonaTraitSelector(results_dir=args.results_dir)

    if args.compare:
        # 比較モード
        selector.compare_selection_methods()
    else:
        # 推奨生成モード
        selector.generate_recommendations(
            selection_method=args.method,
            k=args.k,
            threshold=args.threshold,
            output_file=args.output
        )


if __name__ == "__main__":
    main()
