#!/usr/bin/env python3
"""
新しいペルソナを追加

指定されたエピソードからペルソナを作成:
- profile.json
- train_turns.json
- test_turns.json
- train_data.json (後方互換性)
- test_data.json (後方互換性)
"""
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc
import sys

# Import functions from extract_conversation_turns.py
sys.path.insert(0, str(Path(__file__).parent))
from extract_conversation_turns import (
    extract_all_turns,
    build_persona_turns,
    split_train_test
)

def load_episode_from_dataset(episode_id: str) -> dict:
    """Load episode data from CC dataset"""
    arrow_dir = Path("data/raw/cc/conversationchronicles/train")

    for arrow_file in arrow_dir.glob("*.arrow"):
        with pa.OSFile(str(arrow_file), 'rb') as source:
            try:
                reader = ipc.open_stream(source)
            except:
                source.seek(0)
                reader = ipc.open_file(source)

            batches = [batch for batch in reader]
            if batches:
                table = pa.Table.from_batches(batches)
                data = table.to_pydict()

                for i, data_id in enumerate(data['dataID']):
                    if data_id == episode_id:
                        return {key: values[i] for key, values in data.items()}

    return None

def create_persona(episode_id: str, speaker_suffix: str):
    """
    Create a new persona from episode

    Args:
        episode_id: e.g., "episode-175246"
        speaker_suffix: "A" or "B"
    """
    persona_id = f"{episode_id}_{speaker_suffix}"
    print(f"\n{'='*80}")
    print(f"Creating: {persona_id}")
    print(f"{'='*80}")

    # Load episode data
    episode_data = load_episode_from_dataset(episode_id)
    if episode_data is None:
        print(f"❌ Episode {episode_id} not found in dataset")
        return False

    # Extract all turns
    all_turns = extract_all_turns(episode_data)
    print(f"  Total turns in episode: {len(all_turns)}")

    # Determine speaker
    relationship = episode_data['relationship']
    speakers_in_episode = set(turn["speaker"] for turn in all_turns)
    print(f"  Relationship: {relationship}")
    print(f"  Available speakers: {speakers_in_episode}")

    # Map suffix to actual speaker
    speakers_list = sorted(speakers_in_episode)
    if len(speakers_list) != 2:
        print(f"❌ Expected 2 speakers, found {len(speakers_list)}")
        return False

    target_speaker = speakers_list[0] if speaker_suffix == "A" else speakers_list[1]
    base_speaker = target_speaker.replace(f" {speaker_suffix}", "")  # Remove suffix if present

    print(f"  Target speaker: {target_speaker}")

    # Extract persona turns
    persona_turns = build_persona_turns(all_turns, target_speaker, context_window=3)
    print(f"  Persona turns extracted: {len(persona_turns)}")

    if len(persona_turns) == 0:
        print(f"❌ No turns found")
        return False

    # Split train/test
    train_turns, test_turns = split_train_test(persona_turns, train_ratio=0.5)
    print(f"  Train turns: {len(train_turns)}")
    print(f"  Test turns: {len(test_turns)}")

    # Create output directory
    output_dir = Path("personas_cc") / persona_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create profile.json (minimal version - traits will be computed later)
    profile = {
        "persona_id": persona_id,
        "speaker_role": base_speaker,
        "relationship": relationship,
        "split": "train",
        "num_turns": len(persona_turns),
        "num_turns_train": len(train_turns),
        "num_turns_test": len(test_turns),
        "example_utterances": [turn["ground_truth"] for turn in persona_turns[:10]],
        "traits": {
            "directness": 0.0,
            "emotional_valence": 0.0,
            "social_orientation": 0.0,
            "audience_focus": 0.0,
            "risk_orientation": 0.0
        }
    }

    # Save profile
    with open(output_dir / "profile.json", "w") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    # Save train_turns.json
    train_data = {
        "persona_id": persona_id,
        "speaker_role": base_speaker,
        "relationship": relationship,
        "total_turns": len(train_turns),
        "turns": train_turns
    }
    with open(output_dir / "train_turns.json", "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    # Save test_turns.json
    test_data = {
        "persona_id": persona_id,
        "speaker_role": base_speaker,
        "relationship": relationship,
        "total_turns": len(test_turns),
        "turns": test_turns
    }
    with open(output_dir / "test_turns.json", "w") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    # Create backward-compatible train_data.json and test_data.json
    # (just utterance lists for old code)
    train_data_old = {
        "persona_id": persona_id,
        "speaker_role": base_speaker,
        "relationship": relationship,
        "utterances": [turn["ground_truth"] for turn in train_turns]
    }
    with open(output_dir / "train_data.json", "w") as f:
        json.dump(train_data_old, f, indent=2, ensure_ascii=False)

    test_data_old = {
        "persona_id": persona_id,
        "speaker_role": base_speaker,
        "relationship": relationship,
        "utterances": [turn["ground_truth"] for turn in test_turns]
    }
    with open(output_dir / "test_data.json", "w") as f:
        json.dump(test_data_old, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Created persona: {persona_id}")
    print(f"     {output_dir}/")

    return True

def main():
    # New personas to add
    new_personas = [
        ("episode-175246", "A"),  # Husband, 82 turns
        ("episode-51953", "A"),   # Neighbors A, 82 turns
    ]

    print("="*80)
    print("Adding New Personas")
    print("="*80)

    success_count = 0
    for episode_id, speaker_suffix in new_personas:
        if create_persona(episode_id, speaker_suffix):
            success_count += 1

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully added: {success_count}/{len(new_personas)}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
