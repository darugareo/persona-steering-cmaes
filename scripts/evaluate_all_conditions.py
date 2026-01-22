import os
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import argparse
import time

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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


def build_base_prompt(turn, profile):
    """Base条件のプロンプト（役割明示、スタイルなし）"""

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

Your reply:"""

    return prompt


def build_fewshot_prompt(turn, profile):
    """Few-shot条件のプロンプト（役割明示 + example utterances）"""

    speaker_role = profile["speaker_role"]
    partner_role = get_partner_role(speaker_role)
    relationship_type = get_relationship_type(profile["relationship"])

    # Example utterances を取得（最大5個）
    examples = profile.get("example_utterances", [])[:5]
    examples_text = "\n".join([f"- \"{ex}\"" for ex in examples])

    prompt = f"""You are Speaker A in a conversation with Speaker B.

Speaker A role: {speaker_role}
Speaker B role: {partner_role}
Relationship: {relationship_type}

Your communication style examples:
{examples_text}

Your task:
Given the conversation so far and Speaker B's latest utterance,
produce a single natural reply as Speaker A that matches your communication style.

Constraints:
- Respond only as Speaker A.
- Do not change roles or speak as Speaker B.
- Do not introduce facts not present in the conversation.
- The reply should be natural given the specified relationship.
- Match the communication style shown in the examples.
- Output only the reply text.

Conversation so far:
{turn.get('context', '')}

Speaker B's latest message:
"{turn['user']}"

Your reply:"""

    return prompt


def load_persona_data(persona_id):
    """ペルソナデータを読み込み"""
    persona_dir = Path(f"personas_cc/{persona_id}")

    with open(persona_dir / "profile.json") as f:
        profile = json.load(f)

    test_file = persona_dir / "test_turns_persona_specific.json"
    with open(test_file) as f:
        test_data = json.load(f)

    return profile, test_data["turns"]


def load_trait_vectors(trait_dir="data/steering_vectors_v2", layer=20):
    """Trait Vectorsを読み込み"""
    trait_vectors = []
    for i in range(1, 6):
        trait_file = Path(trait_dir) / f"R{i}" / f"layer{layer}_svd.pt"
        tv = torch.load(trait_file, map_location="cpu", weights_only=False)
        trait_vectors.append(tv)
    return trait_vectors


def generate_random_weights(l2_target=10.0):
    """ランダムな重みを生成（L2ノルム ≈ l2_target）"""
    weights = np.random.randn(5)
    weights = weights / np.linalg.norm(weights) * l2_target
    return weights.tolist()


def generate_response(model, tokenizer, prompt, max_new_tokens=100, device="cuda"):
    """モデルで応答を生成"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()


def generate_with_steering(model, tokenizer, prompt, trait_vectors, weights,
                           alpha=2.0, target_layer=20, max_new_tokens=100, device="cuda"):
    """Steeringを適用して生成"""

    # Steering vectorを計算
    steering_vector = torch.zeros_like(trait_vectors[0])
    for w, tv in zip(weights, trait_vectors):
        steering_vector += w * tv
    steering_vector = steering_vector.to(device)

    # Hook関数
    def steering_hook(module, input, output):
        # Output is tuple: (hidden_states, ...)
        hidden_states = output[0]
        # Reshape steering_vector to (1, 1, hidden_size) for broadcasting
        steering_vec = steering_vector.view(1, 1, -1)
        steered_hidden_states = hidden_states + alpha * steering_vec
        # Return modified output as tuple
        if isinstance(output, tuple):
            return (steered_hidden_states,) + output[1:]
        else:
            return steered_hidden_states

    # Hookを登録
    layer = model.model.layers[target_layer]
    handle = layer.register_forward_hook(steering_hook)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()

    finally:
        handle.remove()


def judge_score(response, ground_truth, profile):
    """GPT-4oでスコアを評価（1-5）"""

    prompt = f"""You are evaluating how well a generated response matches the ground truth.

## Persona
- Role: {profile['speaker_role']}
- Relationship: {profile['relationship']}

## Ground Truth Response
"{ground_truth}"

## Generated Response
"{response}"

## Evaluation Criteria

1. **Style Similarity** (40%)
   - Similar vocabulary, sentence length, formality level
   - Similar emotional expression

2. **Content Relevance** (30%)
   - Addresses the same topic/point
   - Appropriate for the conversation

3. **Persona Consistency** (30%)
   - Speaks as the correct role
   - Maintains relationship dynamics

## Output Format (JSON only)
{{
    "score": 1-5,
    "reason": "Brief explanation (1-2 sentences)"
}}

1 = Very different, 5 = Very similar
Output JSON only."""

    try:
        response_api = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150
        )

        result_text = response_api.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        return json.loads(result_text)
    except Exception as e:
        return {"score": 0, "reason": f"Error: {str(e)}"}


def evaluate_condition(
    condition_name,
    persona_id,
    model,
    tokenizer,
    trait_vectors,
    output_dir,
    alpha=2.0
):
    """1ペルソナ・1条件を評価"""

    profile, test_turns = load_persona_data(persona_id)

    results = []

    for i, turn in enumerate(test_turns):
        print(f"    Turn {i+1}/{len(test_turns)}...", end=" ")

        # プロンプト作成
        if condition_name == "base":
            prompt = build_base_prompt(turn, profile)
            response = generate_response(model, tokenizer, prompt)

        elif condition_name == "random":
            prompt = build_base_prompt(turn, profile)
            random_weights = generate_random_weights(l2_target=10.0)
            response = generate_with_steering(
                model, tokenizer, prompt, trait_vectors,
                random_weights, alpha=alpha
            )

        elif condition_name == "fewshot":
            prompt = build_fewshot_prompt(turn, profile)
            response = generate_response(model, tokenizer, prompt)

        else:
            raise ValueError(f"Unknown condition: {condition_name}")

        # 評価
        judgment = judge_score(response, turn["assistant"], profile)

        results.append({
            "turn_id": i,
            "ground_truth": turn["assistant"],
            "response": response,
            "score": judgment.get("score", 0),
            "reason": judgment.get("reason", "")
        })

        print(f"Score: {judgment.get('score', 'N/A')}")
        time.sleep(0.5)  # Rate limiting

    # 平均スコア
    scores = [r["score"] for r in results if r["score"] > 0]
    avg_score = sum(scores) / len(scores) if scores else 0

    # 保存
    persona_output_dir = Path(output_dir) / condition_name / persona_id
    persona_output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "persona_id": persona_id,
        "condition": condition_name,
        "speaker_role": profile["speaker_role"],
        "relationship": profile["relationship"],
        "num_turns": len(test_turns),
        "average_score": avg_score,
        "results": results
    }

    with open(persona_output_dir / "evaluation.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--trait_dir", default="data/steering_vectors_v2")
    parser.add_argument("--output_dir", default="results/condition_comparison")
    parser.add_argument("--conditions", nargs="+", default=["base", "random", "fewshot"],
                        choices=["base", "random", "fewshot"])
    parser.add_argument("--personas", nargs="+", default=None)
    parser.add_argument("--alpha", type=float, default=2.0)
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

    # Trait Vectors読み込み（Random Steering用）
    print("Loading trait vectors...")
    trait_vectors = load_trait_vectors(args.trait_dir)

    # 対象ペルソナ
    if args.personas:
        target_personas = args.personas
    else:
        target_personas = get_target_personas()

    print(f"Target personas: {len(target_personas)}")
    print(f"Conditions: {args.conditions}")

    # 全結果
    all_results = []

    for condition in args.conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {condition.upper()}")
        print(f"{'='*60}")

        condition_results = []

        for i, persona_id in enumerate(target_personas):
            print(f"\n[{i+1}/{len(target_personas)}] {persona_id}")

            try:
                result = evaluate_condition(
                    condition_name=condition,
                    persona_id=persona_id,
                    model=model,
                    tokenizer=tokenizer,
                    trait_vectors=trait_vectors,
                    output_dir=args.output_dir,
                    alpha=args.alpha
                )
                condition_results.append(result)
                print(f"  Average score: {result['average_score']:.2f}")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                condition_results.append({
                    "persona_id": persona_id,
                    "condition": condition,
                    "error": str(e)
                })

        # 条件ごとのサマリー
        successful = [r for r in condition_results if "average_score" in r]
        if successful:
            avg = sum(r["average_score"] for r in successful) / len(successful)
            print(f"\n📊 {condition.upper()} Average: {avg:.2f}")

        all_results.extend(condition_results)

    # 全体サマリー保存
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # レポート生成
    generate_comparison_report(all_results, output_dir, args.conditions)

    print(f"\n✅ 完了")
    print(f"📁 出力: {output_dir}")


def generate_comparison_report(all_results, output_dir, conditions):
    """比較レポートを生成"""

    report = f"""# Condition Comparison Report

**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary by Condition

| Condition | N | Average Score | Std Dev |
|-----------|---|---------------|---------|
"""

    for condition in conditions:
        cond_results = [r for r in all_results
                       if r.get("condition") == condition and "average_score" in r]
        if cond_results:
            scores = [r["average_score"] for r in cond_results]
            avg = np.mean(scores)
            std = np.std(scores)
            report += f"| {condition} | {len(cond_results)} | {avg:.2f} | {std:.2f} |\n"

    report += "\n## Per-Persona Results\n\n"
    report += "| Persona | Role |"
    for cond in conditions:
        report += f" {cond} |"
    report += "\n|---------|------|"
    for _ in conditions:
        report += "------|"
    report += "\n"

    # ペルソナごとに集計
    personas = set(r["persona_id"] for r in all_results if "persona_id" in r)
    for persona_id in sorted(personas):
        persona_results = {r["condition"]: r for r in all_results
                         if r.get("persona_id") == persona_id and "average_score" in r}
        if persona_results:
            role = list(persona_results.values())[0].get("speaker_role", "N/A")
            report += f"| {persona_id} | {role} |"
            for cond in conditions:
                if cond in persona_results:
                    report += f" {persona_results[cond]['average_score']:.2f} |"
                else:
                    report += " - |"
            report += "\n"

    with open(output_dir / "COMPARISON_REPORT.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
