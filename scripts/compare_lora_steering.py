#!/usr/bin/env python3
# scripts/compare_lora_steering.py

import json
from pathlib import Path

# Load LoRA results
lora_results_file = Path("results/lora_evaluation_response_only/summary.json")
with open(lora_results_file) as f:
    lora_data = json.load(f)

# Load Steering results (sigma5 best scores)
steering_dir = Path("optimization_results_sigma5")

steering_scores = {}
for persona_dir in steering_dir.iterdir():
    if persona_dir.is_dir():
        summary_file = persona_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                data = json.load(f)
                persona_id = data["persona_id"]
                steering_scores[persona_id] = data.get("best_score", 0)

# Match personas
lora_personas = {p["persona_id"]: p for p in lora_data["per_persona"] if "average_score" in p}

# Calculate steering average
common_personas = set(lora_personas.keys()) & set(steering_scores.keys())
steering_avg = sum(steering_scores[p] for p in common_personas) / len(common_personas) if common_personas else 0
lora_avg = lora_data['overall_average_score']

print("# LoRA vs Steering (Sigma=5) Comparison\n")
print("## Overall Results\n")
print(f"| Method | Average Score (1-5) |")
print(f"|--------|---------------------|")
print(f"| **LoRA (Response-only)** | **{lora_avg:.2f}** |")
print(f"| Steering (Sigma=5) | {steering_avg:.2f} |")
print(f"\n**Winner: {'LoRA' if lora_avg > steering_avg else ('Steering' if steering_avg > lora_avg else 'Tie')}**")
print(f"**Difference**: {abs(lora_avg - steering_avg):.2f} points\n")

print("\n## Per-Persona Comparison\n")
print("| Persona | LoRA | Steering | Diff | Winner |")
print("|---------|------|----------|------|--------|")

lora_wins = 0
steering_wins = 0
ties = 0

for persona_id in sorted(common_personas):
    lora_score = lora_personas[persona_id]["average_score"]
    steering_score = steering_scores[persona_id]
    
    diff = lora_score - steering_score
    if abs(diff) < 0.01:
        winner = "Tie"
        ties += 1
    elif diff > 0:
        winner = "LoRA"
        lora_wins += 1
    else:
        winner = "Steering"
        steering_wins += 1
    
    print(f"| {persona_id} | {lora_score:.2f} | {steering_score:.2f} | {diff:+.2f} | {winner} |")

print(f"\n## Summary\n")
print(f"- Total personas: {len(common_personas)}")
print(f"- LoRA wins: {lora_wins}")
print(f"- Steering wins: {steering_wins}")
print(f"- Ties: {ties}")

# Save to markdown
output = f"""# LoRA vs Steering Method Comparison

**Generated**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Results

| Method | Average Score (1-5) | Type |
|--------|--------------------|-|
| **LoRA (Response-only)** | **{lora_avg:.2f}** | Fine-tuning |
| Steering (Sigma=5) | {steering_avg:.2f} | Training-free |

**Winner**: {'✅ LoRA' if lora_avg > steering_avg else ('❌ Steering' if steering_avg > lora_avg else 'Tie')}  
**Difference**: {abs(lora_avg - steering_avg):.2f} points

## Per-Persona Results

| Persona | LoRA | Steering | Diff | Winner |
|---------|------|----------|------|--------|
"""

for persona_id in sorted(common_personas):
    lora_score = lora_personas[persona_id]["average_score"]
    steering_score = steering_scores[persona_id]
    diff = lora_score - steering_score
    
    if abs(diff) < 0.01:
        winner = "Tie"
    elif diff > 0:
        winner = "LoRA"
    else:
        winner = "Steering"
    
    output += f"| {persona_id} | {lora_score:.2f} | {steering_score:.2f} | {diff:+.2f} | {winner} |\n"

output += f"""
## Summary Statistics

- Total personas compared: {len(common_personas)}
- LoRA wins: {lora_wins}
- Steering wins: {steering_wins}
- Ties: {ties}

## Analysis

### Performance
- LoRA average: {lora_avg:.2f}/5.0
- Steering average: {steering_avg:.2f}/5.0
- Performance difference: {abs(lora_avg - steering_avg):.2f} points

### Win Rate
- LoRA win rate: {100*lora_wins/len(common_personas):.1f}%
- Steering win rate: {100*steering_wins/len(common_personas):.1f}%

## Conclusion

{'LoRA fine-tuning shows better personalization performance than the training-free steering method.' if lora_avg > steering_avg else 'The training-free steering method achieves competitive or better results than LoRA fine-tuning.'}
"""

with open("results/LORA_VS_STEERING_COMPARISON.md", "w") as f:
    f.write(output)

print("\n✅ Comparison saved to: results/LORA_VS_STEERING_COMPARISON.md")
