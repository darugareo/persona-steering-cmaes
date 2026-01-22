"""
Phase 2結果の可視化スクリプト

各ペルソナのBase、Prompt Persona、Proposedの応答を比較表示
"""

import json
from pathlib import Path
from typing import Dict, List


def load_judge_logs(persona_id: str, phase: int = 2, experiment: str = "multi_judge", method: str = "proposed") -> List[Dict]:
    """指定されたペルソナ/手法のJSONL judge logsを読み込む"""
    base_dir = Path("reports/raw_judge_logs")
    phase_dir = f"phase{phase}"

    log_dir = base_dir / persona_id / phase_dir / experiment
    if not log_dir.exists():
        print(f"⚠️  ログディレクトリが見つかりません: {log_dir}")
        return []

    records = []
    for path in log_dir.glob(f"{method}_seed*.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    records.append(rec)
                except json.JSONDecodeError:
                    continue

    print(f"✓ {persona_id} ({method}): {len(records)}件のレコード")
    return records


def display_example(persona_id: str, records: List[Dict], idx: int = 0):
    """1つの例を詳細表示"""
    if not records or idx >= len(records):
        print(f"⚠️  レコードがありません (persona={persona_id}, idx={idx})")
        return

    rec = records[idx]
    meta = rec.get("meta", {})

    print("\n" + "="*80)
    print(f"ペルソナ: {persona_id}")
    print(f"手法: {meta.get('method', 'N/A')} | Seed: {meta.get('seed', 'N/A')} | Layer: {meta.get('layer', 'N/A')} | Alpha: {meta.get('alpha', 'N/A')}")
    print("="*80)

    print("\n【プロンプト】")
    print(f"{rec.get('prompt', 'N/A')}")

    print("\n【ベースライン応答】")
    print(f"{rec.get('baseline_response', 'N/A')}")

    print("\n【Proposed（ステアリング）応答】")
    print(f"{rec.get('steered_response', 'N/A')}")

    judge_out = rec.get("judge_output", {})
    if isinstance(judge_out, dict):
        print(f"\n【Judge評価】")
        print(f"Winner: {judge_out.get('winner', 'N/A')}")
        print(f"Confidence: {judge_out.get('confidence', 'N/A')}")
        print(f"Persona Fit Score A: {judge_out.get('persona_fit_score_a', 'N/A')}")
        print(f"Persona Fit Score B: {judge_out.get('persona_fit_score_b', 'N/A')}")
        print(f"説明: {judge_out.get('explanation', 'N/A')}")

    print("\n" + "="*80)


def compare_methods(persona_id: str, phase: int = 2):
    """Base, Prompt Persona, Proposedの3手法を比較"""
    methods = ["base", "prompt_persona", "proposed"]
    all_records = {}

    print(f"\n{'='*80}")
    print(f"ペルソナ: {persona_id} - 3手法の比較")
    print(f"{'='*80}")

    for method in methods:
        records = load_judge_logs(persona_id, phase=phase, method=method)
        all_records[method] = records

    # 各手法の統計
    print(f"\n【統計サマリー】")
    for method, records in all_records.items():
        if records:
            scores_b = []
            wins = 0
            for rec in records:
                judge_out = rec.get("judge_output", {})
                if isinstance(judge_out, dict):
                    if "persona_fit_score_b" in judge_out:
                        scores_b.append(judge_out["persona_fit_score_b"])
                    if judge_out.get("winner") == "B":
                        wins += 1

            if scores_b:
                mean_score = sum(scores_b) / len(scores_b)
                win_rate = (wins / len(records)) * 100
                print(f"  {method:20s}: 平均Persona Fit {mean_score:.2f} | 勝率 {win_rate:.1f}% ({len(records)}件)")
            else:
                print(f"  {method:20s}: スコアなし")
        else:
            print(f"  {method:20s}: データなし")

    # 最初の例を表示
    print(f"\n{'='*80}")
    print(f"【例1: Base vs Proposed】")
    print(f"{'='*80}")

    if all_records["base"] and all_records["proposed"]:
        base_rec = all_records["base"][0]
        prop_rec = all_records["proposed"][0]

        print("\n【プロンプト】")
        print(f"{base_rec.get('prompt', 'N/A')}")

        print("\n【Base応答】")
        print(f"{base_rec.get('baseline_response', 'N/A')[:500]}..." if len(base_rec.get('baseline_response', '')) > 500 else base_rec.get('baseline_response', 'N/A'))

        print("\n【Proposed応答】")
        print(f"{prop_rec.get('steered_response', 'N/A')[:500]}..." if len(prop_rec.get('steered_response', '')) > 500 else prop_rec.get('steered_response', 'N/A'))

        base_judge = base_rec.get("judge_output", {})
        prop_judge = prop_rec.get("judge_output", {})

        print(f"\n【Judge評価比較】")
        print(f"  Base:")
        print(f"    Persona Fit Score B: {base_judge.get('persona_fit_score_b', 'N/A')}")
        print(f"    Winner: {base_judge.get('winner', 'N/A')}")
        print(f"    説明: {base_judge.get('explanation', 'N/A')}")
        print(f"\n  Proposed:")
        print(f"    Persona Fit Score B: {prop_judge.get('persona_fit_score_b', 'N/A')}")
        print(f"    Winner: {prop_judge.get('winner', 'N/A')}")
        print(f"    説明: {prop_judge.get('explanation', 'N/A')}")


def main():
    """全ペルソナの結果を表示"""
    persona_ids = ["episode-184019_A", "episode-239427_A", "episode-118328_B"]

    print("\n" + "="*80)
    print("Phase 2: Multi-Judge評価結果の可視化")
    print("="*80)

    for persona_id in persona_ids:
        compare_methods(persona_id, phase=2)
        print("\n")

    print("\n✓ 可視化完了")


if __name__ == "__main__":
    main()
