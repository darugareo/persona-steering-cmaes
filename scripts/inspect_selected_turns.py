#!/usr/bin/env python3
"""
選定されたペルソナ特有ターンの内容を確認

使用方法:
    python scripts/inspect_selected_turns.py episode-184019_A
"""

import json
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python scripts/inspect_selected_turns.py <persona_id>")
    sys.exit(1)

persona_id = sys.argv[1]
persona_dir = Path(f"personas_cc/{persona_id}")

print(f"=" * 80)
print(f"選定されたターンの確認: {persona_id}")
print(f"=" * 80)

for split in ["train", "test"]:
    file_path = persona_dir / f"{split}_turns_persona_specific.json"

    if not file_path.exists():
        print(f"\n⚠️ {file_path} not found")
        continue

    with open(file_path) as f:
        data = json.load(f)

    print(f"\n{'─'*80}")
    print(f"{split.upper()} Turns")
    print(f"{'─'*80}")
    print(f"Total original turns: {data['total_turns']}")
    print(f"Selected turns: {data['selected_turns']}")
    print(f"Selection rate: {data['selection_rate']*100:.1f}%")

    # 選定されたターンの例を表示
    turns = data['turns']

    print(f"\n📝 Selected Turns:")
    for i, turn in enumerate(turns[:5], 1):  # 最初の5個
        print(f"\n  {i}. Confidence: {turn['specificity_confidence']}/5")
        print(f"     Reason: {turn['specificity_reason']}")
        print(f"     Criteria: {', '.join(turn['criteria_met'])}")
        print(f"     User: {turn['user'][:80]}...")
        print(f"     Assistant: {turn['assistant'][:80]}...")

    if len(turns) > 5:
        print(f"\n  ... and {len(turns) - 5} more turns")

print(f"\n{'='*80}")
print(f"✅ 確認完了")
print(f"{'='*80}")
