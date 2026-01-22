#!/usr/bin/env python3
# scripts/optimize_steering_new.py

import os
import json
import torch
import cma
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import time

# 除外ペルソナ
EXCLUDED_PERSONAS = {
    "episode-204347_A", "episode-225888_A", "episode-239427_A",
    "episode-37624_A", "episode-38144_A", "episode-51953_A", "episode-98947_A"
}


def get_target_personas():
    """対象ペルソナのリストを取得"""
    personas_dir = Path("personas_cc")
    all_personas = [p.name for p in personas_dir.iterdir()
                    if p.is_dir() and p.name.startswith("episode-")]
    return sorted([p for p in all_personas if p not in EXCLUDED_PERSONAS])


def get_partner_role(speaker_role):
    """Partner roleを取得"""
    role_pairs = {
        "Husband": "Wife", "Wife": "Husband",
        "Parent": "Child", "Child": "Parent",
        "Mentor": "Mentee", "Mentee": "Mentor",
    }
    return role_pairs.get(speaker_role, "Partner")


def get_relationship_type(relationship):
    """Relationship typeを正規化"""
    relationship_map = {
        "Husband and Wife": "married couple",
        "Parent and Child": "parent-child",
        "Mentee and Mentor": "mentor-mentee",
        "Classmates": "classmates",
        "Neighbors": "neighbors",
    }
    return relationship_map.get(relationship, relationship.lower())


def build_prompt(turn, profile):
    """新しいプロンプト構造（役割明示、スタイルなし）"""

    speaker_role = profile["speaker_role"]
    partner_role = get_partner_role(speaker_role)
    relationship_type = get_relationship_type(profile["relationship"])

    prompt = f"""You are Speaker A in a conversation with Speaker B.

Speaker A role: {speaker_role}
Speaker B role: {partner_role}
Relationship: {relationship_type}

Your task:
Given the conversation so far and Speaker B's latest utterance,
produce a single natural reply as Speaker A.

Constraints:
- Respond only as Speaker A.
- Do not change roles or speak as Speaker B.
- Do not introduce facts not present in the conversation.
- The reply should be natural given the specified relationship.
- Output only the reply text.

Conversation so far:
{turn.get('context', '')}

Speaker B's latest message:
"{turn['user']}"

Your reply: """

    return prompt


def load_persona_data(persona_id):
    """ペルソナデータを読み込み"""
    persona_dir = Path(f"personas_cc/{persona_id}")

    with open(persona_dir / "profile.json") as f:
        profile = json.load(f)

    train_file = persona_dir / "train_turns_persona_specific.json"
    with open(train_file) as f:
        train_data = json.load(f)

    return profile, train_data["turns"]


def load_trait_vectors(trait_dir="data/steering_vectors_v2", layer=20):
    """Trait Vectorsを読み込み"""
    trait_vectors = {}
    for i in range(1, 6):  # R1-R5
        trait_name = f"R{i}"
        trait_file = Path(trait_dir) / trait_name / f"layer{layer}_svd.pt"
        if not trait_file.exists():
            raise FileNotFoundError(f"Trait vector not found: {trait_file}")
        trait_vectors[trait_name] = torch.load(trait_file)
    return trait_vectors


class SteeringOptimizer:
    """CMA-ESによるSteering重み最適化"""

    def __init__(
        self,
        model,
        tokenizer,
        trait_vectors,
        target_layer=20,
        alpha=2.0,
        device="cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.trait_vectors = trait_vectors
        self.target_layer = target_layer
        self.alpha = alpha
        self.device = device

        # Hookのための変数
        self.steering_vector = None
        self.hook_handle = None

    def _steering_hook(self, module, input, output):
        """Steeringを適用するフック"""
        if self.steering_vector is None:
            return output

        # Output is tuple: (hidden_states, *optional_outputs)
        hidden_states = output[0]  # Shape: (batch_size, seq_len, hidden_size)

        # Add steering vector (broadcast across batch and sequence)
        steering_vec = self.steering_vector.to(
            hidden_states.device, dtype=hidden_states.dtype
        )
        # Reshape to (1, 1, hidden_size) for proper broadcasting
        steering_vec = steering_vec.view(1, 1, -1)
        steered_hidden_states = hidden_states + self.alpha * steering_vec

        # Return modified output
        if isinstance(output, tuple):
            return (steered_hidden_states,) + output[1:]
        else:
            return steered_hidden_states

    def compute_steering_vector(self, weights):
        """重みからSteering Vectorを計算"""
        sv = torch.zeros_like(self.trait_vectors["R1"])
        for i, trait in enumerate(["R1", "R2", "R3", "R4", "R5"]):
            sv += weights[i] * self.trait_vectors[trait]
        return sv

    def generate_with_steering(self, prompt, weights, max_new_tokens=100):
        """Steeringを適用して生成"""

        # Steering vectorを計算
        self.steering_vector = self.compute_steering_vector(weights)

        # Hookを登録
        layer = self.model.model.layers[self.target_layer]
        self.hook_handle = layer.register_forward_hook(self._steering_hook)

        try:
            # トークナイズ
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # デコード（プロンプト部分を除く）
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            return response.strip()

        finally:
            # Hookを解除
            if self.hook_handle is not None:
                self.hook_handle.remove()
            self.steering_vector = None

    def compute_style_similarity(self, generated, ground_truth):
        """
        スタイル類似度を計算（簡易版）

        本番ではGPT-4o Judgeを使用するが、
        最適化ループでは高速な簡易指標を使用
        """
        # 文長の類似度
        len_ratio = min(len(generated), len(ground_truth)) / max(len(generated), len(ground_truth), 1)

        # 単語の重複率
        gen_words = set(generated.lower().split())
        gt_words = set(ground_truth.lower().split())
        if len(gen_words) == 0 or len(gt_words) == 0:
            word_overlap = 0
        else:
            word_overlap = len(gen_words & gt_words) / len(gen_words | gt_words)

        # 句読点の使用率
        gen_punct = sum(1 for c in generated if c in "!?.,;:")
        gt_punct = sum(1 for c in ground_truth if c in "!?.,;:")
        punct_sim = 1 - min(abs(gen_punct - gt_punct) / max(gt_punct, 1), 1)

        # 総合スコア（0-1）
        score = 0.3 * len_ratio + 0.5 * word_overlap + 0.2 * punct_sim
        return score

    def evaluate_weights(self, weights, profile, train_turns, num_samples=None):
        """重みを評価（平均Fitness）"""

        if num_samples is not None:
            turns = train_turns[:num_samples]
        else:
            turns = train_turns

        scores = []

        for turn in turns:
            prompt = build_prompt(turn, profile)
            response = self.generate_with_steering(prompt, weights)
            score = self.compute_style_similarity(response, turn["assistant"])
            scores.append(score)

        return sum(scores) / len(scores)


def optimize_persona(
    persona_id,
    optimizer,
    sigma=5.0,
    max_generations=15,
    population_size=8,
    output_dir="optimization_results_new"
):
    """1ペルソナを最適化"""

    print(f"\n{'='*60}")
    print(f"Optimizing: {persona_id}")
    print(f"{'='*60}")

    profile, train_turns = load_persona_data(persona_id)

    print(f"  Speaker role: {profile['speaker_role']}")
    print(f"  Relationship: {profile['relationship']}")
    print(f"  Train turns: {len(train_turns)}")
    print(f"  σ={sigma}, α={optimizer.alpha}")

    # CMA-ES初期化
    es = cma.CMAEvolutionStrategy(
        x0=[0.0] * 5,
        sigma0=sigma,
        inopts={
            'popsize': population_size,
            'maxiter': max_generations,
            'verb_disp': 1,
            'verb_log': 0
        }
    )

    best_weights = None
    best_fitness = -float('inf')
    history = []

    start_time = time.time()
    generation = 0

    while not es.stop():
        generation += 1
        solutions = es.ask()

        fitnesses = []
        for i, weights in enumerate(solutions):
            fitness = optimizer.evaluate_weights(weights, profile, train_turns)
            fitnesses.append(-fitness)  # CMA-ESは最小化

            print(f"    Gen {generation}, Individual {i+1}/{len(solutions)}: fitness={fitness:.4f}")

        es.tell(solutions, fitnesses)

        # Best更新
        gen_best_idx = np.argmin(fitnesses)
        gen_best_fitness = -fitnesses[gen_best_idx]
        gen_best_weights = solutions[gen_best_idx]

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_weights = gen_best_weights.copy()

        history.append({
            "generation": generation,
            "best_fitness": gen_best_fitness,
            "mean_fitness": -np.mean(fitnesses),
            "best_weights": gen_best_weights.tolist()
        })

        print(f"  Gen {generation}: best={gen_best_fitness:.4f}, mean={-np.mean(fitnesses):.4f}")

    elapsed_time = time.time() - start_time

    # L2ノルム計算
    l2_norm = np.sqrt(sum(w**2 for w in best_weights))

    print(f"\n  ✅ Optimization complete")
    print(f"  Best fitness: {best_fitness:.4f}")
    print(f"  L2 norm: {l2_norm:.2f}")
    print(f"  Time: {elapsed_time/60:.1f} minutes")

    # 結果を保存
    result = {
        "persona_id": persona_id,
        "speaker_role": profile["speaker_role"],
        "relationship": profile["relationship"],
        "num_train_turns": len(train_turns),
        "best_weights": {f"R{i+1}": float(w) for i, w in enumerate(best_weights)},
        "best_fitness": float(best_fitness),
        "l2_norm": float(l2_norm),
        "settings": {
            "sigma": sigma,
            "alpha": optimizer.alpha,
            "max_generations": max_generations,
            "population_size": population_size
        },
        "elapsed_time_seconds": elapsed_time,
        "history": history
    }

    persona_output_dir = Path(output_dir) / persona_id
    persona_output_dir.mkdir(parents=True, exist_ok=True)

    with open(persona_output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    # 重みのみを別ファイルに保存（後で使いやすいように）
    with open(persona_output_dir / "best_weights.json", "w") as f:
        json.dump({f"R{i+1}": float(w) for i, w in enumerate(best_weights)}, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--trait_dir", default="data/steering_vectors_v2")
    parser.add_argument("--output_dir", default="optimization_results_new")
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--max_generations", type=int, default=15)
    parser.add_argument("--population_size", type=int, default=8)
    parser.add_argument("--target_layer", type=int, default=20)
    parser.add_argument("--personas", nargs="+", default=None)
    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # モデル読み込み
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    # Trait Vectors読み込み
    print("Loading trait vectors...")
    trait_vectors = load_trait_vectors(args.trait_dir, args.target_layer)
    print(f"  Loaded {len(trait_vectors)} trait vectors")

    # Optimizer作成
    optimizer = SteeringOptimizer(
        model=model,
        tokenizer=tokenizer,
        trait_vectors=trait_vectors,
        target_layer=args.target_layer,
        alpha=args.alpha,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 対象ペルソナ
    if args.personas:
        target_personas = args.personas
    else:
        target_personas = get_target_personas()

    print(f"Target personas: {len(target_personas)}")
    print(f"Settings: σ={args.sigma}, α={args.alpha}")

    # 各ペルソナを最適化
    results = []

    for i, persona_id in enumerate(target_personas):
        print(f"\n[{i+1}/{len(target_personas)}] {persona_id}")

        try:
            result = optimize_persona(
                persona_id=persona_id,
                optimizer=optimizer,
                sigma=args.sigma,
                max_generations=args.max_generations,
                population_size=args.population_size,
                output_dir=args.output_dir
            )
            results.append(result)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "persona_id": persona_id,
                "error": str(e)
            })

    # 全体サマリー保存
    with open(output_dir / "optimization_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # レポート生成
    generate_optimization_report(results, output_dir, args)

    print(f"\n{'='*60}")
    print(f"✅ 完了: {len([r for r in results if 'error' not in r])}/{len(results)} personas optimized")
    print(f"📁 出力: {output_dir}")


def generate_optimization_report(results, output_dir, args):
    """最適化レポートを生成"""

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    total_time = sum(r.get("elapsed_time_seconds", 0) for r in successful)

    if successful:
        l2_norms = [r["l2_norm"] for r in successful]
        fitnesses = [r["best_fitness"] for r in successful]
    else:
        l2_norms = []
        fitnesses = []

    report = f"""# Steering Optimization Report (New Prompt Structure)

**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Configuration

- σ (sigma): {args.sigma}
- α (alpha): {args.alpha}
- Max generations: {args.max_generations}
- Population size: {args.population_size}
- Target layer: {args.target_layer}
- Prompt: Role-specified, no style examples

## Summary

- Total personas: {len(results)}
- Successful: {len(successful)}
- Failed: {len(failed)}
- Total time: {total_time/3600:.1f} hours
"""

    if l2_norms:
        report += f"""
## L2 Norm Statistics

- Mean: {np.mean(l2_norms):.2f}
- Median: {np.median(l2_norms):.2f}
- Min: {np.min(l2_norms):.2f}
- Max: {np.max(l2_norms):.2f}

## Fitness Statistics

- Mean: {np.mean(fitnesses):.4f}
- Median: {np.median(fitnesses):.4f}
- Min: {np.min(fitnesses):.4f}
- Max: {np.max(fitnesses):.4f}
"""

    report += """
## Per-Persona Results

| Persona | Role | L2 Norm | Fitness | Time (min) |
|---------|------|---------|---------|------------|
"""

    for r in sorted(successful, key=lambda x: x["l2_norm"], reverse=True):
        report += f"| {r['persona_id']} | {r['speaker_role']} | {r['l2_norm']:.2f} | {r['best_fitness']:.4f} | {r['elapsed_time_seconds']/60:.1f} |\n"

    if failed:
        report += f"\n## Failed Personas\n\n"
        for r in failed:
            report += f"- {r['persona_id']}: {r['error']}\n"

    if successful:
        report += f"""
## Best Weights Examples

### Top 3 by L2 Norm
"""

        for r in sorted(successful, key=lambda x: x["l2_norm"], reverse=True)[:3]:
            report += f"\n**{r['persona_id']}** (L2={r['l2_norm']:.2f})\n```\n"
            for trait, weight in r["best_weights"].items():
                report += f"  {trait}: {weight:+.3f}\n"
            report += "```\n"

    with open(output_dir / "OPTIMIZATION_REPORT.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
