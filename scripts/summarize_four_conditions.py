#!/usr/bin/env python3
"""
4条件比較結果の集計とレポート生成
"""
import json
from pathlib import Path
from typing import Dict, List

def calculate_win_rate(results: List[Dict]) -> Dict:
    """Calculate win rates from comparison results"""
    wins = {}
    total = len(results)
    errors = 0

    for r in results:
        winner = r["winner"]
        if winner == "error":
            errors += 1
        else:
            wins[winner] = wins.get(winner, 0) + 1

    valid_total = total - errors

    win_rates = {}
    for condition, count in wins.items():
        win_rates[condition] = count / valid_total if valid_total > 0 else 0.0

    return {
        "wins": wins,
        "total": total,
        "errors": errors,
        "valid_total": valid_total,
        "win_rates": win_rates
    }

def load_all_results():
    """Load all persona results"""
    comp_dir = Path("results/four_conditions/comparisons")

    personas = set()
    for file in comp_dir.glob("*.json"):
        persona_id = file.stem.split("_steering_")[0].split("_prompt_")[0].split("_hybrid_")[0]
        personas.add(persona_id)

    all_results = {}

    for persona_id in personas:
        all_results[persona_id] = {}

        for comp_name in ["steering_vs_base", "prompt_vs_steering", "hybrid_vs_prompt"]:
            file = comp_dir / f"{persona_id}_{comp_name}.json"
            if file.exists():
                with open(file) as f:
                    data = json.load(f)
                    all_results[persona_id][comp_name] = data["results"]

    return all_results

def generate_report():
    """Generate summary markdown report"""
    all_results = load_all_results()

    if not all_results:
        print("No results found!")
        return

    # Aggregate across personas
    aggregated = {
        "steering_vs_base": [],
        "prompt_vs_steering": [],
        "hybrid_vs_prompt": []
    }

    for persona_id, persona_results in all_results.items():
        for comp_name, comp_results in persona_results.items():
            aggregated[comp_name].extend(comp_results)

    # Calculate overall win rates
    overall_stats = {}
    for comp_name, comp_results in aggregated.items():
        overall_stats[comp_name] = calculate_win_rate(comp_results)

    # Generate markdown report
    report = "# 4条件比較実験 - 結果レポート\n\n"
    report += f"**Generated**: 2026-01-06\n"
    report += f"**Personas**: {len(all_results)}\n"
    report += f"**Turns per persona**: 10\n"
    report += f"**Total judgments**: {sum(s['total'] for s in overall_stats.values())}\n\n"

    report += "---\n\n"
    report += "## Overall Win Rates\n\n"

    # Steering vs Base
    stats_1 = overall_stats["steering_vs_base"]
    steering_wins = stats_1["wins"].get("steering", 0)
    base_wins = stats_1["wins"].get("base", 0)
    steering_rate = stats_1["win_rates"].get("steering", 0.0) * 100

    report += f"### 1. Steering vs Base\n\n"
    report += f"**Winner**: {'**Steering**' if steering_wins > base_wins else '**Base**'}\n\n"
    report += f"| Condition | Wins | Win Rate |\n"
    report += f"|-----------|------|----------|\n"
    report += f"| Steering  | {steering_wins}/{stats_1['valid_total']} | {steering_rate:.1f}% |\n"
    report += f"| Base      | {base_wins}/{stats_1['valid_total']} | {100-steering_rate:.1f}% |\n\n"

    if stats_1["errors"] > 0:
        report += f"*Errors: {stats_1['errors']}/{stats_1['total']}*\n\n"

    # Prompt vs Steering
    stats_2 = overall_stats["prompt_vs_steering"]
    prompt_wins = stats_2["wins"].get("prompt", 0)
    steering_wins_2 = stats_2["wins"].get("steering", 0)
    prompt_rate = stats_2["win_rates"].get("prompt", 0.0) * 100

    report += f"### 2. Prompt vs Steering\n\n"
    report += f"**Winner**: {'**Prompt**' if prompt_wins > steering_wins_2 else '**Steering**'}\n\n"
    report += f"| Condition | Wins | Win Rate |\n"
    report += f"|-----------|------|----------|\n"
    report += f"| Prompt    | {prompt_wins}/{stats_2['valid_total']} | {prompt_rate:.1f}% |\n"
    report += f"| Steering  | {steering_wins_2}/{stats_2['valid_total']} | {100-prompt_rate:.1f}% |\n\n"

    if stats_2["errors"] > 0:
        report += f"*Errors: {stats_2['errors']}/{stats_2['total']}*\n\n"

    # Hybrid vs Prompt
    stats_3 = overall_stats["hybrid_vs_prompt"]
    hybrid_wins = stats_3["wins"].get("hybrid", 0)
    prompt_wins_3 = stats_3["wins"].get("prompt", 0)
    hybrid_rate = stats_3["win_rates"].get("hybrid", 0.0) * 100

    report += f"### 3. Hybrid vs Prompt\n\n"
    report += f"**Winner**: {'**Hybrid**' if hybrid_wins > prompt_wins_3 else '**Prompt**'}\n\n"
    report += f"| Condition | Wins | Win Rate |\n"
    report += f"|-----------|------|----------|\n"
    report += f"| Hybrid    | {hybrid_wins}/{stats_3['valid_total']} | {hybrid_rate:.1f}% |\n"
    report += f"| Prompt    | {prompt_wins_3}/{stats_3['valid_total']} | {100-hybrid_rate:.1f}% |\n\n"

    if stats_3["errors"] > 0:
        report += f"*Errors: {stats_3['errors']}/{stats_3['total']}*\n\n"

    report += "---\n\n"
    report += "## Per-Persona Results\n\n"

    for persona_id, persona_results in sorted(all_results.items()):
        report += f"### {persona_id}\n\n"

        for comp_name in ["steering_vs_base", "prompt_vs_steering", "hybrid_vs_prompt"]:
            if comp_name not in persona_results:
                continue

            stats = calculate_win_rate(persona_results[comp_name])

            comp_display = comp_name.replace("_", " ").title()
            report += f"**{comp_display}**:\n"

            for condition, rate in stats["win_rates"].items():
                wins = stats["wins"][condition]
                report += f"- {condition.title()}: {wins}/{stats['valid_total']} ({rate*100:.1f}%)\n"

            report += "\n"

    report += "---\n\n"
    report += "## Interpretation\n\n"

    # Add interpretation based on results
    report += "### Key Findings\n\n"

    if steering_rate > 60:
        report += f"1. **Steering is effective**: {steering_rate:.1f}% win rate against Base\n"
    else:
        report += f"1. **Steering has limited effect**: {steering_rate:.1f}% win rate against Base\n"

    if prompt_rate > 60:
        report += f"2. **Prompt engineering is more powerful than Steering**: {prompt_rate:.1f}% win rate\n"
    elif prompt_rate < 40:
        report += f"2. **Steering outperforms Prompt engineering**: {100-prompt_rate:.1f}% win rate\n"
    else:
        report += f"2. **Prompt and Steering are comparable**: Prompt {prompt_rate:.1f}% vs Steering {100-prompt_rate:.1f}%\n"

    if hybrid_rate > 60:
        report += f"3. **Combining both methods improves performance**: {hybrid_rate:.1f}% win rate\n"
    elif hybrid_rate < 40:
        report += f"3. **Adding Steering to Prompt does not help**: Hybrid only {hybrid_rate:.1f}% win rate\n"
    else:
        report += f"3. **Hybrid and Prompt are similar**: No clear benefit from combining both\n"

    report += "\n---\n\n"
    report += "## Data Files\n\n"
    report += "- Generations: `results/four_conditions/generations/`\n"
    report += "- Comparisons: `results/four_conditions/comparisons/`\n"

    # Save report
    output_path = Path("results/four_conditions/SUMMARY.md")
    with open(output_path, "w") as f:
        f.write(report)

    print(f"✅ Report saved to: {output_path}")
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Steering vs Base: {steering_rate:.1f}% (Steering wins)")
    print(f"Prompt vs Steering: {prompt_rate:.1f}% (Prompt wins)")
    print(f"Hybrid vs Prompt: {hybrid_rate:.1f}% (Hybrid wins)")
    print("="*80)

if __name__ == "__main__":
    generate_report()
