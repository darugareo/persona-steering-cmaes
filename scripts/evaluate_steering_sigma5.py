#!/usr/bin/env python3
# scripts/evaluate_steering_sigma5.py

import os
import json
import torch
from pathlib import Path
from openai import OpenAI
import argparse
from persona_opt.internal_steering_l3 import Llama3ActivationSteerer

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EXCLUDED_PERSONAS = {
    "episode-204347_A", "episode-225888_A", "episode-239427_A",
    "episode-37624_A", "episode-38144_A", "episode-51953_A", "episode-98947_A"
}


def load_trait_vectors(target_layer=20):
    """Load trait vectors from data/steering_vectors_v2/"""
    trait_vectors = {}
    for trait in ["R1", "R2", "R3", "R4", "R5"]:
        vector_path = f"data/steering_vectors_v2/{trait}/layer{target_layer}_svd.pt"
        trait_vectors[trait] = torch.load(vector_path, weights_only=True)
    return trait_vectors


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


def load_test_data(persona_id):
    persona_dir = Path(f"personas_cc/{persona_id}")

    with open(persona_dir / "profile.json") as f:
        profile = json.load(f)

    test_file = persona_dir / "test_turns_persona_specific.json"
    with open(test_file) as f:
        test_data = json.load(f)

    return profile, test_data["turns"]


def load_steering_config(persona_id, sigma5_dir):
    # Try summary.json first, then result.json
    summary_file = Path(sigma5_dir) / persona_id / "summary.json"
    result_file = Path(sigma5_dir) / persona_id / "result.json"

    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
    elif result_file.exists():
        with open(result_file) as f:
            summary = json.load(f)
        # Extract alpha from settings if available
        if "settings" in summary:
            summary["alpha"] = summary["settings"].get("alpha", 2.0)
    else:
        raise FileNotFoundError(f"Neither summary.json nor result.json found for {persona_id}")

    weights = summary["best_weights"]
    alpha = summary.get("alpha", 2.0)
    return [weights[f"R{i+1}"] for i in range(5)], alpha


def generate_steering_response(model, prompt, weights, alpha, max_new_tokens=150):
    # Convert weights to trait_weights dict
    trait_weights = {f"R{i+1}": weights[i] for i in range(len(weights))}

    # Register hooks with multi-trait steering
    model.register_hooks(
        multi_trait_vectors=model.trait_vectors,  # Assume trait vectors are loaded
        trait_weights=trait_weights
    )

    # Generate
    response = model.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )

    # Remove hooks
    model.remove_hooks()

    return response.strip()


def judge_similarity(response, ground_truth, profile):
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


def evaluate_persona(persona_id, model, sigma5_dir):
    print(f"\n{'='*60}")
    print(f"Evaluating: {persona_id}")
    print(f"{'='*60}")

    # Load steering config
    weights, alpha = load_steering_config(persona_id, sigma5_dir)
    print(f"  Steering config loaded (alpha={alpha})")

    # Load test data
    profile, test_turns = load_test_data(persona_id)
    print(f"  Test turns: {len(test_turns)}")

    results = []

    for i, turn in enumerate(test_turns):
        print(f"  Turn {i+1}/{len(test_turns)}...", end=" ", flush=True)

        prompt = build_prompt(turn, profile)

        # Generate with steering
        response = generate_steering_response(model, prompt, weights, alpha)

        # Evaluate similarity
        judgment = judge_similarity(response, turn["assistant"], profile)

        results.append({
            "turn_id": i,
            "ground_truth": turn["assistant"],
            "steering_response": response,
            "score": judgment.get("score", 0),
            "judgment": judgment
        })

        print(f"Score: {judgment.get('score', 'N/A')}")

    # Calculate average
    scores = [r["score"] for r in results if r["score"] > 0]
    avg_score = sum(scores) / len(scores) if scores else 0

    print(f"  Average score: {avg_score:.2f}")

    # Save results
    output_dir = Path(f"results/steering_evaluation_sigma5/{persona_id}")
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

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--sigma5_dir", default="optimization_results_sigma5")
    parser.add_argument("--output_dir", default="results/steering_evaluation_sigma5")
    parser.add_argument("--personas", nargs="+", default=None)
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model = Llama3ActivationSteerer(model_name=args.model_name, device="cuda")

    # Load trait vectors
    print("Loading trait vectors...")
    model.trait_vectors = load_trait_vectors(target_layer=model.target_layer)
    print(f"Trait vectors loaded: {list(model.trait_vectors.keys())}")

    # Get target personas
    if args.personas:
        target_personas = args.personas
    else:
        target_personas = get_target_personas()

    print(f"Target personas: {len(target_personas)}")

    # Evaluate
    all_results = []

    for i, persona_id in enumerate(target_personas):
        print(f"\n[{i+1}/{len(target_personas)}] {persona_id}")

        try:
            result = evaluate_persona(persona_id, model, args.sigma5_dir)
            all_results.append(result)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_results.append({
                "persona_id": persona_id,
                "error": str(e)
            })

    # Overall summary
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Generate report
    generate_evaluation_report(overall_summary, output_dir)

    print(f"\n✅ 完了")
    print(f"📊 Overall average score: {overall_summary['overall_average_score']:.2f}")


def generate_evaluation_report(summary, output_dir):
    report = f"""# Steering Method Evaluation Report (Sigma=5)

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
