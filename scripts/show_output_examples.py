#!/usr/bin/env python3
"""
Base vs Steering 実際の出力例を表示

ノートブックで確認する前に、コマンドラインで代表例をクイック確認
"""

import json
from pathlib import Path
from collections import defaultdict


def print_comparison(result, title=""):
    """Print a single comparison nicely formatted"""
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
    print("="*80)

    print(f"\n📋 Persona: {result['persona_id']}")
    print(f"🏆 Winner: {result['winner'].upper()}")
    print(f"📊 Confidence: {result['confidence']}/5.0")
    print(f"⚖️  Judge: {result['judge_winner']} (Steering={'A' if result['steering_is_a'] else 'B'})")

    print(f"\n📝 Prompt:")
    print(f"   {result['prompt']}")

    print(f"\n🔵 BASE Response (No Steering):")
    print("-" * 80)
    print(result['response_base'])
    print("-" * 80)

    print(f"\n🔴 STEERING Response (With Optimized Vectors):")
    print("-" * 80)
    print(result['response_steering'])
    print("-" * 80)

    print(f"\n🧑‍⚖️ Judge Explanation:")
    print("-" * 80)
    print(result['explanation'])
    print("-" * 80)


def main():
    # Load data
    results_file = Path("results/base_vs_steering/comparison_results.json")

    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return

    with open(results_file) as f:
        data = json.load(f)

    results = data["results"]

    print("="*80)
    print("BASE vs STEERING: 実際の出力例検査")
    print("="*80)
    print(f"\nTotal comparisons: {len(results)}")
    print(f"Config: {data['config']}")

    # Count outcomes
    winner_counts = {"tie": 0, "steering": 0, "base": 0}
    for r in results:
        winner_counts[r["winner"]] += 1

    print(f"\nOutcome distribution:")
    for winner, count in winner_counts.items():
        print(f"  {winner}: {count} ({100*count/len(results):.1f}%)")

    # Separate by outcome
    ties = [r for r in results if r["winner"] == "tie"]
    steering_wins = [r for r in results if r["winner"] == "steering"]
    base_wins = [r for r in results if r["winner"] == "base"]

    # ============================================================
    # 1. TIE EXAMPLES (most common)
    # ============================================================
    print("\n\n" + "🟡"*40)
    print("🟡 PART 1: TIE EXAMPLES (引き分け - 最も多い)")
    print("🟡"*40)

    print(f"\n引き分けケース数: {len(ties)} ({100*len(ties)/len(results):.1f}%)")
    print("上位3例を表示:")

    for i, result in enumerate(ties[:3], 1):
        print_comparison(result, title=f"TIE Example #{i}")

    # ============================================================
    # 2. STEERING WIN EXAMPLES
    # ============================================================
    print("\n\n" + "🟢"*40)
    print("🟢 PART 2: STEERING WIN EXAMPLES (Steering勝利)")
    print("🟢"*40)

    print(f"\nSteering勝利数: {len(steering_wins)} ({100*len(steering_wins)/len(results):.1f}%)")
    print("全例を表示:")

    for i, result in enumerate(steering_wins[:10], 1):  # Limit to 10 for readability
        print_comparison(result, title=f"STEERING WIN Example #{i}")

    if len(steering_wins) > 10:
        print(f"\n... and {len(steering_wins)-10} more steering wins")

    # ============================================================
    # 3. BASE WIN EXAMPLES
    # ============================================================
    print("\n\n" + "🔴"*40)
    print("🔴 PART 3: BASE WIN EXAMPLES (Base勝利 - Steeringが逆効果)")
    print("🔴"*40)

    print(f"\nBase勝利数: {len(base_wins)} ({100*len(base_wins)/len(results):.1f}%)")
    print("全例を表示:")

    for i, result in enumerate(base_wins, 1):
        print_comparison(result, title=f"BASE WIN Example #{i}")

    # ============================================================
    # 4. PERSONA-SPECIFIC ANALYSIS
    # ============================================================
    print("\n\n" + "📊"*40)
    print("📊 PART 4: ペルソナ別分析")
    print("📊"*40)

    # Group by persona
    persona_outcomes = defaultdict(lambda: {"tie": 0, "steering": 0, "base": 0})

    for r in results:
        persona_id = r["persona_id"]
        winner = r["winner"]
        persona_outcomes[persona_id][winner] += 1

    # Find effective personas
    effective_personas = []
    for persona_id, outcomes in persona_outcomes.items():
        total = sum(outcomes.values())
        if outcomes["tie"] < total:  # Not all ties
            effective_personas.append(persona_id)

    print(f"\n効果があったペルソナ数: {len(effective_personas)}")
    print(f"効果がなかったペルソナ数: {len(persona_outcomes) - len(effective_personas)}")

    # Show details for effective personas
    print("\n効果があったペルソナの詳細:")
    for persona_id in sorted(effective_personas):
        outcomes = persona_outcomes[persona_id]
        total = sum(outcomes.values())
        print(f"\n  {persona_id}:")
        print(f"    Total: {total}")
        print(f"    Tie: {outcomes['tie']} ({100*outcomes['tie']/total:.1f}%)")
        print(f"    Steering wins: {outcomes['steering']} ({100*outcomes['steering']/total:.1f}%)")
        print(f"    Base wins: {outcomes['base']} ({100*outcomes['base']/total:.1f}%)")

        # Show one example from this persona
        persona_results = [r for r in results if r["persona_id"] == persona_id]

        # Prefer steering win example if available
        steering_examples = [r for r in persona_results if r["winner"] == "steering"]
        if steering_examples:
            print(f"\n    代表例（Steering勝利）:")
            print_comparison(steering_examples[0], title=f"{persona_id} Example")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n\n" + "="*80)
    print("📈 SUMMARY")
    print("="*80)

    print(f"""
主な発見:
1. 引き分けが{100*len(ties)/len(results):.1f}%を占める
   → BaseとSteeringの出力がほぼ同じ

2. Decisive comparisons（引き分け以外）では:
   - Steering勝利: {len(steering_wins)}/{len(steering_wins)+len(base_wins)} ({100*len(steering_wins)/(len(steering_wins)+len(base_wins)):.1f}%)
   - Base勝利: {len(base_wins)}/{len(steering_wins)+len(base_wins)} ({100*len(base_wins)/(len(steering_wins)+len(base_wins)):.1f}%)

3. ペルソナ別:
   - 効果あり: {len(effective_personas)}/{len(persona_outcomes)} ({100*len(effective_personas)/len(persona_outcomes):.1f}%)
   - 効果なし（全て引き分け）: {len(persona_outcomes)-len(effective_personas)}/{len(persona_outcomes)} ({100*(len(persona_outcomes)-len(effective_personas))/len(persona_outcomes):.1f}%)

推奨:
- Jupyter notebook で視覚的に確認: notebooks/inspect_base_vs_steering_outputs.ipynb
- 詳細分析: notebooks/analyze_base_vs_steering_executed.ipynb
""")


if __name__ == "__main__":
    main()
