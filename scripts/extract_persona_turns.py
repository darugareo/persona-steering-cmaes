#!/usr/bin/env python3
"""
Conversation Chroniclesデータから26ペルソナのターン数を抽出

pyarrowを使わずに、直接パース処理を実装
"""

import json
import struct
from pathlib import Path
from collections import defaultdict

# 26ペルソナのリスト
PERSONA_IDS = [
    "episode-184019_A", "episode-239427_A", "episode-118328_B",
    "episode-148668_B", "episode-87126_A", "episode-134226_A",
    "episode-179307_A", "episode-98323_A", "episode-173843_A",
    "episode-225888_A", "episode-204347_A", "episode-29916_A",
    "episode-7102_A", "episode-31569_B", "episode-37985_A",
    "episode-228221_A", "episode-98947_A", "episode-158821_B",
    "episode-137872_B", "episode-132229_B", "episode-24857_A",
    "episode-38144_A", "episode-238726_A", "episode-198972_A",
    "episode-84804_A", "episode-136741_A"
]


def parse_persona_id(full_id):
    """episode-XXXXX_A -> (episode-XXXXX, A)"""
    parts = full_id.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None


def try_datasets_load():
    """Try loading with datasets library with workarounds"""
    try:
        # Disable Arrow backend temporarily
        import os
        os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

        from datasets import load_dataset

        print("Loading Conversation Chronicles with HuggingFace datasets...")
        dataset = load_dataset(
            "arrow",
            data_files={
                "train": "data/raw/cc/conversationchronicles/train/data-*.arrow"
            }
        )

        print(f"Loaded {len(dataset['train'])} episodes")

        # Parse personas
        persona_lookup = {}
        for pid in PERSONA_IDS:
            episode_id, speaker = parse_persona_id(pid)
            if episode_id and speaker:
                persona_lookup[pid] = {"episode": episode_id, "speaker": speaker, "turns": []}

        print(f"\nProcessing {len(dataset['train'])} episodes...")
        for idx, example in enumerate(dataset['train']):
            if idx % 10000 == 0:
                print(f"  Processed {idx:,} episodes...")

            episode_id = example.get("dataID", "")

            # Check if this episode is one of our personas
            for pid, info in persona_lookup.items():
                if episode_id == info["episode"]:
                    # Count turns for this speaker across all 5 sessions
                    total_turns_A = 0
                    total_turns_B = 0

                    # Process each session
                    for session_name in ['first', 'second', 'third', 'fourth', 'fifth']:
                        speakers_key = f"{session_name}_session_speakers"
                        speakers = example.get(speakers_key, [])

                        # Count turns for speaker A and B
                        for speaker in speakers:
                            if speaker == 'A':
                                total_turns_A += 1
                            elif speaker == 'B':
                                total_turns_B += 1

                    # Store the turn count for the appropriate speaker
                    if info["speaker"] == "A" and total_turns_A > 0:
                        info["turns"].append(total_turns_A)
                    elif info["speaker"] == "B" and total_turns_B > 0:
                        info["turns"].append(total_turns_B)

        return persona_lookup

    except Exception as e:
        print(f"datasets load failed: {e}")
        return None


def main():
    print("="*90)
    print("26ペルソナの会話ターン数抽出")
    print("="*90)

    # Try datasets library approach
    persona_data = try_datasets_load()

    if persona_data is None:
        print("\nデータ読み込みに失敗しました。")
        print("環境の制約により、完全なターン数統計は取得できません。")
        return

    # Print results
    print("\n" + "="*90)
    print("結果サマリー")
    print("="*90)

    print(f"\n{'Persona ID':<25} {'会話数':<10} {'平均ターン':<12} {'最小':<8} {'最大':<8} {'総ターン'}")
    print("-" * 90)

    all_stats = []
    for pid in PERSONA_IDS:
        if pid in persona_data and persona_data[pid]["turns"]:
            turns_list = persona_data[pid]["turns"]
            num_convs = len(turns_list)
            avg_turns = sum(turns_list) / num_convs
            min_turns = min(turns_list)
            max_turns = max(turns_list)
            total_turns = sum(turns_list)

            print(f"{pid:<25} {num_convs:<10} {avg_turns:<12.1f} {min_turns:<8} {max_turns:<8} {total_turns}")
            all_stats.append({
                "persona_id": pid,
                "num_conversations": num_convs,
                "avg_turns": avg_turns,
                "min_turns": min_turns,
                "max_turns": max_turns,
                "total_turns": total_turns,
                "turns_list": turns_list
            })
        else:
            print(f"{pid:<25} データなし")

    print("="*90)

    # Overall statistics
    if all_stats:
        total_convs = sum(s["num_conversations"] for s in all_stats)
        total_turns = sum(s["total_turns"] for s in all_stats)
        all_turns = [t for s in all_stats for t in s["turns_list"]]

        print(f"\n全体統計:")
        print(f"  データ取得成功ペルソナ数: {len(all_stats)}/{len(PERSONA_IDS)}")
        print(f"  総会話数: {total_convs:,}")
        print(f"  総ターン数: {total_turns:,}")
        print(f"  平均ターン数/会話: {total_turns/total_convs:.1f}")
        print(f"  最小ターン数: {min(all_turns)}")
        print(f"  最大ターン数: {max(all_turns)}")

    # Save to JSON
    output_file = Path("results/persona_turn_statistics.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "personas": all_stats,
            "summary": {
                "total_conversations": total_convs if all_stats else 0,
                "total_turns": total_turns if all_stats else 0,
                "num_personas_with_data": len(all_stats),
                "total_personas": len(PERSONA_IDS)
            }
        }, f, indent=2)

    print(f"\n詳細データを保存: {output_file}")


if __name__ == "__main__":
    main()
