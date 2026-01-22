#!/usr/bin/env python3
# scripts/evaluate_lora.py

import os
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from openai import OpenAI
import argparse

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EXCLUDED_PERSONAS = {
    "episode-204347_A", "episode-225888_A", "episode-239427_A",
    "episode-37624_A", "episode-38144_A", "episode-51953_A", "episode-98947_A"
}


def get_target_personas():
    personas_dir = Path("personas_cc")
    all_personas = [p.name for p in personas_dir.iterdir()
                    if p.is_dir() and p.name.startswith("episode-")]
    return sorted([p for p in all_personas if p not in EXCLUDED_PERSONAS])


def get_partner_role(speaker_role):
    role_pairs = {
        "Husband": "Wife", "Wife": "Husband",
        "Parent": "Child", "Child": "Parent",
        "Mentor": "Mentee", "Mentee": "Mentor",
    }
    return role_pairs.get(speaker_role, "Partner")


def get_relationship_type(relationship):
    relationship_map = {
        "Husband and Wife": "married couple",
        "Parent and Child": "parent-child",
        "Mentee and Mentor": "mentor-mentee",
        "Classmates": "classmates",
        "Neighbors": "neighbors",
    }
    return relationship_map.get(relationship, relationship.lower())


def build_prompt(turn, profile):
    """新しいプロンプト構造でプロンプトを生成"""
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


def load_test_data(persona_id):
    """テストデータを読み込み"""
    persona_dir = Path(f"personas_cc/{persona_id}")

    with open(persona_dir / "profile.json") as f:
        profile = json.load(f)

    test_file = persona_dir / "test_turns_persona_specific.json"
    with open(test_file) as f:
        test_data = json.load(f)

    return profile, test_data["turns"]


def generate_response(model, tokenizer, prompt, max_new_tokens=150):
    """モデルで応答を生成"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def judge_similarity(response, ground_truth, profile):
    """GPT-4oでGround Truthとの類似度を評価"""

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
    "style_match": "low/medium/high",
    "content_match": "low/medium/high",
    "persona_match": "low/medium/high",
    "reason": "Brief explanation"
}}

1 = Very different, 5 = Very similar
Output JSON only."""

    try:
        response_api = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200
        )

        result_text = response_api.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        return json.loads(result_text)
    except Exception as e:
        return {"score": 0, "reason": f"Error: {str(e)}"}


def evaluate_persona(persona_id, base_model, tokenizer, lora_dir):
    """1ペルソナを評価"""

    print(f"\n{'='*60}")
    print(f"Evaluating: {persona_id}")
    print(f"{'='*60}")

    # LoRA重みを読み込み
    lora_path = Path(lora_dir) / persona_id / "lora_weights"
    if not lora_path.exists():
        print(f"  ❌ LoRA weights not found: {lora_path}")
        return None

    peft_model = PeftModel.from_pretrained(base_model, str(lora_path))
    peft_model.eval()

    # テストデータ読み込み
    profile, test_turns = load_test_data(persona_id)
    print(f"  Test turns: {len(test_turns)}")

    results = []

    for i, turn in enumerate(test_turns):
        print(f"  Turn {i+1}/{len(test_turns)}...", end=" ")

        prompt = build_prompt(turn, profile)

        # LoRAモデルで生成
        response = generate_response(peft_model, tokenizer, prompt)

        # 類似度評価
        judgment = judge_similarity(response, turn["assistant"], profile)

        results.append({
            "turn_id": i,
            "ground_truth": turn["assistant"],
            "lora_response": response,
            "score": judgment.get("score", 0),
            "judgment": judgment
        })

        print(f"Score: {judgment.get('score', 'N/A')}")

    # 平均スコア
    scores = [r["score"] for r in results if r["score"] > 0]
    avg_score = sum(scores) / len(scores) if scores else 0

    print(f"  Average score: {avg_score:.2f}")

    # 結果を保存
    output_dir = Path(f"results/lora_evaluation/{persona_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "persona_id": persona_id,
        "speaker_role": profile["speaker_role"],
        "relationship": profile["relationship"],
        "num_test_turns": len(test_turns),
        "average_score": avg_score,
        "results": results
    }

    with open(output_dir / "evaluation.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # メモリ解放
    del peft_model
    torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--lora_dir", default="lora_models")
    parser.add_argument("--output_dir", default="results/lora_evaluation")
    parser.add_argument("--personas", nargs="+", default=None)
    args = parser.parse_args()

    # モデル読み込み
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 対象ペルソナ
    if args.personas:
        target_personas = args.personas
    else:
        target_personas = get_target_personas()

    print(f"Target personas: {len(target_personas)}")

    # 評価
    all_results = []

    for i, persona_id in enumerate(target_personas):
        print(f"\n[{i+1}/{len(target_personas)}] {persona_id}")

        try:
            result = evaluate_persona(
                persona_id, base_model, tokenizer, args.lora_dir
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_results.append({
                "persona_id": persona_id,
                "error": str(e)
            })

    # 全体サマリー
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 統計
    successful = [r for r in all_results if "average_score" in r]
    avg_scores = [r["average_score"] for r in successful]

    overall_summary = {
        "total_personas": len(target_personas),
        "successful": len(successful),
        "overall_average_score": sum(avg_scores) / len(avg_scores) if avg_scores else 0,
        "per_persona": all_results
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(overall_summary, f, indent=2)

    # レポート生成
    generate_evaluation_report(overall_summary, output_dir)

    print(f"\n✅ 完了")
    print(f"📊 Overall average score: {overall_summary['overall_average_score']:.2f}")


def generate_evaluation_report(summary, output_dir):
    """評価レポートを生成"""

    report = f"""# LoRA Evaluation Report

**Generated**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- Total personas: {summary['total_personas']}
- Successful evaluations: {summary['successful']}
- **Overall average score: {summary['overall_average_score']:.2f} / 5.0**

## Per-Persona Results

| Persona | Role | Score | Status |
|---------|------|-------|--------|
"""

    for r in summary["per_persona"]:
        if "error" in r:
            report += f"| {r['persona_id']} | - | - | ❌ |\n"
        else:
            report += f"| {r['persona_id']} | {r.get('speaker_role', 'N/A')} | {r['average_score']:.2f} | ✅ |\n"

    with open(output_dir / "EVALUATION_REPORT.md", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
