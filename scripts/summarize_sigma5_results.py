#!/usr/bin/env python3
"""
Summarize σ=5.0 optimization results for 28 personas
"""
import json
from pathlib import Path

results_dir = Path("optimization_results_sigma5")

results = []
for persona_dir in sorted(results_dir.glob("episode-*")):
    summary_file = persona_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            data = json.load(f)
            results.append({
                "persona_id": data["persona_id"],
                "best_score": data["best_score"],
                "l2_norm": data["l2_norm"],
                "iterations": data["iterations"],
                "duration_minutes": data["duration_seconds"] / 60
            })

# Summary statistics
total = len(results)
best_scores = [r["best_score"] for r in results]
l2_norms = [r["l2_norm"] for r in results]
durations = [r["duration_minutes"] for r in results]

perfect_score = sum(1 for s in best_scores if s == 5.0)
high_l2 = sum(1 for l2 in l2_norms if l2 > 5.0)

print("=" * 80)
print("σ=5.0 Optimization Results - 28 Personas")
print("=" * 80)
print(f"\n📊 Overall Statistics:")
print(f"  Total personas: {total}")
print(f"  Perfect score (5.0): {perfect_score}/{total} ({perfect_score/total*100:.1f}%)")
print(f"  L2 norm > 5.0: {high_l2}/{total} ({high_l2/total*100:.1f}%)")
print(f"\n  Best Score:")
print(f"    Min: {min(best_scores):.4f}")
print(f"    Max: {max(best_scores):.4f}")
print(f"    Mean: {sum(best_scores)/len(best_scores):.4f}")
print(f"\n  L2 Norm:")
print(f"    Min: {min(l2_norms):.4f}")
print(f"    Max: {max(l2_norms):.4f}")
print(f"    Mean: {sum(l2_norms)/len(l2_norms):.4f}")
print(f"\n  Duration:")
print(f"    Min: {min(durations):.1f} min")
print(f"    Max: {max(durations):.1f} min")
print(f"    Mean: {sum(durations)/len(durations):.1f} min")
print(f"    Total: {sum(durations)/60:.1f} hours")

print("\n" + "=" * 80)
print("Individual Results:")
print("=" * 80)
print(f"{'Persona':<20} {'Score':>8} {'L2 Norm':>10} {'Iterations':>12} {'Duration':>12}")
print("-" * 80)

for r in results:
    status = "✅" if r["best_score"] == 5.0 and r["l2_norm"] > 5.0 else "⚠️"
    print(f"{r['persona_id']:<20} {r['best_score']:>8.4f} {r['l2_norm']:>10.4f} {r['iterations']:>12} {r['duration_minutes']:>11.1f}m {status}")

print("=" * 80)

# L2 norm distribution
print("\n📊 L2 Norm Distribution:")
ranges = [
    (0, 5, "< 5.0 (Too weak)"),
    (5, 10, "5-10 (Acceptable)"),
    (10, 15, "10-15 (Good)"),
    (15, 100, "> 15 (Strong)")
]

for low, high, label in ranges:
    count = sum(1 for l2 in l2_norms if low <= l2 < high)
    print(f"  {label:<25} {count:>3} ({count/total*100:>5.1f}%)")

# Success criteria: best_score == 5.0 AND l2_norm > 5.0
success = sum(1 for r in results if r["best_score"] == 5.0 and r["l2_norm"] > 5.0)
print(f"\n✅ Success Rate (Score=5.0 AND L2>5.0): {success}/{total} ({success/total*100:.1f}%)")
