#!/usr/bin/env python3
"""
α × σ 調整実験

目的: CMA-ESのα（steering強度）とσ（初期探索幅）を変化させ、
      L2ノルムとSteering効果への影響を検証

実験対象: episode-184019_A（最も効果があったペルソナ）
4条件: (α, σ) = (2.0, 2.0), (5.0, 2.0), (2.0, 5.0), (5.0, 5.0)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.cmaes_persona_optimizer import CMAESPersonaOptimizer
from persona_opt.internal_steering_l3 import Llama3ActivationSteerer
from persona_opt.persona_judge_evaluator import evaluate_with_persona_judge


# 実験条件
CONDITIONS = [
    {"name": "baseline", "alpha": 2.0, "sigma": 2.0, "description": "Current setting (baseline)"},
    {"name": "high_alpha", "alpha": 5.0, "sigma": 2.0, "description": "High α only (stronger steering)"},
    {"name": "high_sigma", "alpha": 2.0, "sigma": 5.0, "description": "High σ only (wider exploration)"},
    {"name": "high_both", "alpha": 5.0, "sigma": 5.0, "description": "Both high (strong steering + wide exploration)"},
]

PERSONA_ID = "episode-184019_A"
DEVICE = "cuda:0"


class AlphaSigmaExperiment:
    """α×σ調整実験"""

    def __init__(self, persona_id: str, device: str = "cuda:0"):
        self.persona_id = persona_id
        self.device = device

        # Load eval prompts (use selected test turns)
        test_turns_path = Path(f"personas_cc/{persona_id}/test_turns_selected.json")
        if test_turns_path.exists():
            with open(test_turns_path) as f:
                data = json.load(f)
                self.eval_turns = data["turns"]
        else:
            # Fallback: use test_turns.json
            test_turns_path = Path(f"personas_cc/{persona_id}/test_turns.json")
            with open(test_turns_path) as f:
                data = json.load(f)
                self.eval_turns = data["turns"][:10]  # Use first 10

        print(f"✓ Loaded {len(self.eval_turns)} evaluation turns")

        # Prepare eval prompts
        self.eval_prompts = []
        for turn in self.eval_turns:
            context = turn["context"]
            input_text = turn["input"]
            prompt = f"""{context}
{input_text}

Response:"""
            self.eval_prompts.append(prompt)

    def run_optimization(self, alpha: float, sigma: float, condition_name: str) -> Dict:
        """
        1条件の最適化を実行

        Args:
            alpha: Steering強度
            sigma: CMA-ES初期探索幅
            condition_name: 条件名

        Returns:
            最適化結果
        """
        print(f"\n{'='*80}")
        print(f"Condition: {condition_name}")
        print(f"  α (alpha): {alpha}")
        print(f"  σ (sigma): {sigma}")
        print(f"{'='*80}\n")

        # Initialize optimizer
        optimizer = CMAESPersonaOptimizer(
            persona_id=self.persona_id,
            layer=20,
            trait_vector_dir="data/steering_vectors_v2",
            persona_dir="personas",
            eval_prompts=self.eval_prompts[:3],  # Use 3 prompts for optimization (faster)
            alpha=alpha
        )

        # Run optimization
        result = optimizer.optimize(
            sigma0=sigma,
            max_iterations=15,  # 15世代
            population_size=8,
            save_dir=f"results/alpha_sigma_experiment/{condition_name}"
        )

        # Calculate L2 norm
        weights = result["best_weights"]
        l2_norm = np.sqrt(sum(w**2 for w in weights.values()))

        print(f"\n✅ Optimization complete:")
        print(f"  Best weights: {weights}")
        print(f"  L2 norm: {l2_norm:.3f}")
        print(f"  Best score: {result['best_score']:.4f}")

        return {
            "condition": condition_name,
            "alpha": alpha,
            "sigma": sigma,
            "best_weights": weights,
            "l2_norm": float(l2_norm),
            "best_score": float(result["best_score"]),
            "iterations": result.get("iterations", 15)
        }

    def evaluate_steering_effect(
        self,
        weights: Dict[str, float],
        alpha: float,
        condition_name: str
    ) -> Dict:
        """
        最適化された重みでBase vs Steering比較

        Args:
            weights: 最適化された重み
            alpha: Steering強度
            condition_name: 条件名

        Returns:
            評価結果
        """
        print(f"\n{'='*80}")
        print(f"Evaluating Steering Effect: {condition_name}")
        print(f"{'='*80}\n")

        # Initialize steerer
        steerer = Llama3ActivationSteerer(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            target_layer=20,
            device=self.device
        )

        # Load trait vectors
        trait_vectors = {}
        trait_names = ["R1", "R2", "R3", "R4", "R5"]
        for trait in trait_names:
            vector_path = Path(f"data/steering_vectors_v2/{trait}/layer20_svd.pt")
            import torch
            trait_vectors[trait] = torch.load(vector_path, map_location='cpu').to(self.device)

        # Generate responses
        results = []
        steering_wins = 0
        base_wins = 0
        ties = 0

        for i, turn in enumerate(self.eval_turns, 1):
            context = turn["context"]
            input_text = turn["input"]

            # Build prompt
            prompt = f"""Continue this conversation naturally.

Conversation so far:
{context}

Partner: {input_text}

You:"""

            # Base response
            steerer.remove_hooks()
            response_base = steerer.generate(
                prompt=prompt,
                max_new_tokens=100,
                temperature=0.0,
                do_sample=False
            )

            # Steering response
            # Build steering vector
            steering_vec = sum(
                weights[trait] * trait_vectors[trait]
                for trait in trait_names
            )
            steerer.register_hooks(steering_vector=steering_vec, alpha=alpha)
            response_steering = steerer.generate(
                prompt=prompt,
                max_new_tokens=100,
                temperature=0.0,
                do_sample=False
            )
            steerer.remove_hooks()

            # Judge evaluation
            import random
            steering_is_a = random.choice([True, False])

            if steering_is_a:
                response_a = response_steering
                response_b = response_base
            else:
                response_a = response_base
                response_b = response_steering

            judge_result = evaluate_with_persona_judge(
                persona_id=self.persona_id,
                prompt=input_text,
                response_a=response_a,
                response_b=response_b,
                trait_name="Overall Persona Fit",
                trait_direction="matches persona style and values",
                base_dir="personas",
                model="gpt-4o",
                temperature=0.3,
                save_raw_log=False
            )

            # Map winner
            judge_winner = judge_result.get("winner", "tie")

            if judge_winner == "A":
                winner = "steering" if steering_is_a else "base"
            elif judge_winner == "B":
                winner = "base" if steering_is_a else "steering"
            else:
                winner = "tie"

            if winner == "steering":
                steering_wins += 1
            elif winner == "base":
                base_wins += 1
            else:
                ties += 1

            print(f"  Turn {i}/{len(self.eval_turns)}: {winner} (confidence: {judge_result.get('confidence', 0)})")

            results.append({
                "turn_id": i,
                "winner": winner,
                "confidence": judge_result.get("confidence", 0),
                "response_base": response_base,
                "response_steering": response_steering
            })

        # Calculate metrics
        total = len(self.eval_turns)
        decisive = steering_wins + base_wins

        steering_win_rate = steering_wins / total if total > 0 else 0.0
        decisive_steering_rate = steering_wins / decisive if decisive > 0 else 0.0

        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"  Steering wins: {steering_wins}/{total} ({100*steering_win_rate:.1f}%)")
        print(f"  Base wins: {base_wins}/{total} ({100*base_wins/total:.1f}%)")
        print(f"  Ties: {ties}/{total} ({100*ties/total:.1f}%)")
        if decisive > 0:
            print(f"  Decisive Steering rate: {100*decisive_steering_rate:.1f}%")
        print(f"{'='*60}\n")

        return {
            "steering_wins": steering_wins,
            "base_wins": base_wins,
            "ties": ties,
            "total": total,
            "steering_win_rate": float(steering_win_rate),
            "decisive_steering_rate": float(decisive_steering_rate) if decisive > 0 else 0.0,
            "details": results
        }

    def run_full_experiment(self, output_dir: Path):
        """
        全条件の実験を実行

        Args:
            output_dir: 出力ディレクトリ
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        all_results = []

        for condition in CONDITIONS:
            print(f"\n\n{'#'*80}")
            print(f"# Condition: {condition['name']}")
            print(f"# {condition['description']}")
            print(f"{'#'*80}\n")

            # Optimization
            opt_result = self.run_optimization(
                alpha=condition["alpha"],
                sigma=condition["sigma"],
                condition_name=condition["name"]
            )

            # Evaluation
            eval_result = self.evaluate_steering_effect(
                weights=opt_result["best_weights"],
                alpha=condition["alpha"],
                condition_name=condition["name"]
            )

            # Combine results
            result = {
                **condition,
                **opt_result,
                "evaluation": eval_result,
                "timestamp": datetime.now().isoformat()
            }

            all_results.append(result)

            # Save intermediate results
            with open(output_dir / "results.json", "w") as f:
                json.dump(all_results, f, indent=2)

            print(f"\n✅ Condition '{condition['name']}' complete")
            print(f"   L2 Norm: {opt_result['l2_norm']:.3f}")
            print(f"   Steering Win Rate: {eval_result['steering_win_rate']:.1%}")

        # Generate summary
        self.generate_summary(all_results, output_dir)

        print(f"\n{'='*80}")
        print("✅ ALL EXPERIMENTS COMPLETE")
        print(f"{'='*80}\n")
        print(f"Results saved to: {output_dir}")

    def generate_summary(self, results: List[Dict], output_dir: Path):
        """結果サマリーを生成"""
        summary_md = f"""# α × σ 調整実験結果

**実施日**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**ペルソナ**: {self.persona_id}
**評価ターン数**: {len(self.eval_turns)}

---

## 条件別結果

| 条件 | α | σ | L2ノルム | Opt Score | Steering勝率 | Decisive勝率 | 引き分け率 |
|------|---|---|---------|-----------|-------------|-------------|----------|
"""

        for result in results:
            eval_res = result["evaluation"]
            summary_md += f"| {result['name']} | {result['alpha']} | {result['sigma']} | "
            summary_md += f"{result['l2_norm']:.3f} | {result['best_score']:.4f} | "
            summary_md += f"{eval_res['steering_win_rate']:.1%} | "
            summary_md += f"{eval_res['decisive_steering_rate']:.1%} | "
            summary_md += f"{eval_res['ties']/eval_res['total']:.1%} |\n"

        # Find best condition
        best_by_l2 = max(results, key=lambda r: r["l2_norm"])
        best_by_winrate = max(results, key=lambda r: r["evaluation"]["steering_win_rate"])

        summary_md += f"""

---

## 主な発見

### L2ノルム
- **最大**: {best_by_l2['name']} (α={best_by_l2['alpha']}, σ={best_by_l2['sigma']}): **{best_by_l2['l2_norm']:.3f}**
- **最小**: {min(results, key=lambda r: r['l2_norm'])['name']}: {min(results, key=lambda r: r['l2_norm'])['l2_norm']:.3f}
- **範囲**: {max(r['l2_norm'] for r in results) - min(r['l2_norm'] for r in results):.3f}

### Steering効果
- **最高勝率**: {best_by_winrate['name']} (α={best_by_winrate['alpha']}, σ={best_by_winrate['sigma']}): **{best_by_winrate['evaluation']['steering_win_rate']:.1%}**
- **最低勝率**: {min(results, key=lambda r: r['evaluation']['steering_win_rate'])['name']}: {min(results, key=lambda r: r['evaluation']['steering_win_rate'])['steering_win_rate']:.1%}

---

## 結論

### αの影響
"""

        # Compare alpha=2.0 vs 5.0 (with same sigma)
        baseline = next(r for r in results if r["name"] == "baseline")
        high_alpha = next(r for r in results if r["name"] == "high_alpha")

        summary_md += f"""
- L2ノルム変化: {baseline['l2_norm']:.3f} → {high_alpha['l2_norm']:.3f} ({((high_alpha['l2_norm']/baseline['l2_norm'])-1)*100:+.1f}%)
- 勝率変化: {baseline['evaluation']['steering_win_rate']:.1%} → {high_alpha['evaluation']['steering_win_rate']:.1%}
- **結論**: {'αを上げるとL2ノルムが増加' if high_alpha['l2_norm'] > baseline['l2_norm'] else 'αを上げてもL2ノルムは変わらない'}

### σの影響
"""

        high_sigma = next(r for r in results if r["name"] == "high_sigma")

        summary_md += f"""
- L2ノルム変化: {baseline['l2_norm']:.3f} → {high_sigma['l2_norm']:.3f} ({((high_sigma['l2_norm']/baseline['l2_norm'])-1)*100:+.1f}%)
- 勝率変化: {baseline['evaluation']['steering_win_rate']:.1%} → {high_sigma['evaluation']['steering_win_rate']:.1%}
- **結論**: {'σを上げると探索範囲が広がりL2ノルムが増加' if high_sigma['l2_norm'] > baseline['l2_norm'] else 'σを上げても大きな変化なし'}

### 推奨設定
- **L2ノルム最大化**: {best_by_l2['name']} (α={best_by_l2['alpha']}, σ={best_by_l2['sigma']})
- **Steering効果最大化**: {best_by_winrate['name']} (α={best_by_winrate['alpha']}, σ={best_by_winrate['sigma']})

---

## 次のステップ

"""

        if best_by_l2["l2_norm"] > 5.0:
            summary_md += "✅ **成功**: L2 > 5.0 達成。この設定で26ペルソナに拡大\n"
        else:
            summary_md += "⚠️ **部分的成功**: L2ノルムは増加したが目標未達。さらに調整が必要\n"

        if best_by_winrate["evaluation"]["steering_win_rate"] > 0.5:
            summary_md += "✅ **Steering効果**: 50%以上の勝率達成\n"
        else:
            summary_md += "⚠️ **要改善**: Steering勝率が50%未満。別の最適化手法も検討\n"

        # Save summary
        with open(output_dir / "SUMMARY.md", "w") as f:
            f.write(summary_md)

        print(f"✓ Saved summary: {output_dir / 'SUMMARY.md'}")


def main():
    parser = argparse.ArgumentParser(description="α × σ adjustment experiment")
    parser.add_argument("--persona_id", default=PERSONA_ID, help="Persona ID")
    parser.add_argument("--device", default=DEVICE, help="Device (cuda:0, cuda:1, etc.)")
    parser.add_argument("--output_dir", default="results/alpha_sigma_experiment", help="Output directory")

    args = parser.parse_args()

    # Run experiment
    experiment = AlphaSigmaExperiment(
        persona_id=args.persona_id,
        device=args.device
    )

    output_dir = Path(args.output_dir)

    experiment.run_full_experiment(output_dir)


if __name__ == "__main__":
    main()
