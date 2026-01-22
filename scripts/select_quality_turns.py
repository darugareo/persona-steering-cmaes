#!/usr/bin/env python3
"""
Test turns品質選定: GPT-4oで評価に適したターンを自動選定
"""
import json
import os
from pathlib import Path
from typing import Dict, List
from openai import OpenAI
import time

class TurnQualitySelector:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def score_turn_quality(self, turn: Dict, persona_profile: Dict) -> Dict:
        """GPT-4oでターンの品質をスコアリング"""
        prompt = f"""You are evaluating the quality of a conversation turn for persona evaluation.

## Persona
- Role: {persona_profile.get("speaker_role", "Unknown")}
- Relationship: {persona_profile.get("relationship", "Unknown")}

## Conversation Turn
Context:
{turn["context"]}

Partner's message:
{turn["input"]}

Persona's response (Ground Truth):
{turn["ground_truth"]}

## Evaluation Criteria
Rate each criterion from 1-5:

1. **Continuity** (1-5): Does the response logically follow from the context and partner's message?
   - 1: Completely unrelated, sudden topic change
   - 3: Somewhat related but awkward transition
   - 5: Perfectly natural continuation

2. **Informativeness** (1-5): Does the response contain substantial content?
   - 1: Just "Yes/No/OK" or very short
   - 3: Brief but meaningful
   - 5: Rich, detailed response

3. **Persona Expression** (1-5): Does the response show the persona's characteristics?
   - 1: Generic, could be anyone
   - 3: Some personality shown
   - 5: Clearly distinctive persona style

Output JSON only:
{{"continuity": X, "informativeness": X, "persona_expression": X, "total": X, "include": true/false, "reason": "brief reason"}}

Set include=true if total >= 10 (out of 15)."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return result
            else:
                print(f"Warning: Could not parse JSON from response: {content}")
                return {
                    "continuity": 3,
                    "informativeness": 3,
                    "persona_expression": 3,
                    "total": 9,
                    "include": False,
                    "reason": "Parse error"
                }

        except Exception as e:
            print(f"Error scoring turn: {e}")
            return {
                "continuity": 3,
                "informativeness": 3,
                "persona_expression": 3,
                "total": 9,
                "include": False,
                "reason": f"Error: {str(e)}"
            }

    def select_turns_for_persona(
        self,
        persona_id: str,
        min_turns: int = 5,
        target_turns: int = 10
    ) -> Dict:
        """1ペルソナのテストターンを選定"""
        print(f"\n{'='*80}")
        print(f"Selecting quality turns: {persona_id}")
        print(f"{'='*80}")

        # Load data
        profile_path = Path(f"personas_cc/{persona_id}/profile.json")
        test_path = Path(f"personas_cc/{persona_id}/test_turns.json")

        with open(profile_path) as f:
            profile = json.load(f)

        with open(test_path) as f:
            test_data = json.load(f)
            test_turns = test_data["turns"]

        print(f"Total test turns: {len(test_turns)}")

        # Score each turn
        scored_turns = []
        for i, turn in enumerate(test_turns):
            print(f"  Scoring turn {i+1}/{len(test_turns)}...", end="\r")

            score = self.score_turn_quality(turn, profile)

            scored_turns.append({
                **turn,
                "quality_scores": score
            })

            # Rate limiting
            time.sleep(0.5)

        print(f"\n  Scoring complete!")

        # Sort by total score
        scored_turns.sort(key=lambda x: x["quality_scores"]["total"], reverse=True)

        # Select top turns
        selected_turns = []
        threshold = 10  # Initial threshold

        while len(selected_turns) < min_turns and threshold > 5:
            selected_turns = [
                t for t in scored_turns
                if t["quality_scores"]["total"] >= threshold
            ]
            if len(selected_turns) < min_turns:
                threshold -= 1

        # If still not enough, take top N
        if len(selected_turns) < min_turns:
            selected_turns = scored_turns[:min_turns]

        # Limit to target
        selected_turns = selected_turns[:target_turns]

        print(f"\n  Selected: {len(selected_turns)}/{len(test_turns)} turns")
        print(f"  Quality threshold: {threshold}/15")
        print(f"  Avg score: {sum(t['quality_scores']['total'] for t in selected_turns) / len(selected_turns):.2f}")

        return {
            "persona_id": persona_id,
            "selection_criteria": f"total_score >= {threshold}",
            "total_original": len(test_turns),
            "total_selected": len(selected_turns),
            "avg_quality_score": sum(t["quality_scores"]["total"] for t in selected_turns) / len(selected_turns),
            "turns": selected_turns
        }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona_ids", nargs="+", required=True)
    parser.add_argument("--min_turns", type=int, default=5)
    parser.add_argument("--target_turns", type=int, default=10)
    args = parser.parse_args()

    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    selector = TurnQualitySelector(api_key)

    # Process each persona
    results = []
    for persona_id in args.persona_ids:
        try:
            result = selector.select_turns_for_persona(
                persona_id,
                min_turns=args.min_turns,
                target_turns=args.target_turns
            )
            results.append(result)

            # Save selected turns
            output_dir = Path(f"personas_cc/{persona_id}")
            output_path = output_dir / "test_turns_selected.json"

            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            print(f"  ✅ Saved to: {output_path}")

        except Exception as e:
            print(f"  ❌ Error processing {persona_id}: {e}")

    # Generate summary
    if results:
        summary_dir = Path("results/data_selection")
        summary_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "total_personas": len(results),
            "avg_selected_per_persona": sum(r["total_selected"] for r in results) / len(results),
            "avg_quality_score": sum(r["avg_quality_score"] for r in results) / len(results),
            "selection_rate": f"{sum(r['total_selected'] for r in results) / sum(r['total_original'] for r in results) * 100:.1f}%",
            "personas": [
                {
                    "persona_id": r["persona_id"],
                    "selected": r["total_selected"],
                    "original": r["total_original"],
                    "avg_score": r["avg_quality_score"]
                }
                for r in results
            ]
        }

        summary_path = summary_dir / "selection_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Personas processed: {summary['total_personas']}")
        print(f"Avg selected per persona: {summary['avg_selected_per_persona']:.1f}")
        print(f"Avg quality score: {summary['avg_quality_score']:.2f}/15")
        print(f"Selection rate: {summary['selection_rate']}")
        print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()
