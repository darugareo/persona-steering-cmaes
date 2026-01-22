#!/usr/bin/env python3
"""
Conversation Chroniclesから文脈付きターン構造を抽出

目的:
- 各ペルソナについて、会話の文脈を含むターンを抽出
- Train/Test分割して保存

出力:
- personas_cc/{persona_id}/train_turns.json
- personas_cc/{persona_id}/test_turns.json
"""
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc
from typing import List, Dict

def load_arrow_data(arrow_dir: Path) -> Dict:
    """Load all arrow files and combine into single dict"""
    print(f"Loading Arrow files from: {arrow_dir}")

    all_data = {}
    arrow_files = sorted(arrow_dir.glob("*.arrow"))

    for arrow_file in arrow_files:
        print(f"  Reading: {arrow_file.name}")
        with pa.OSFile(str(arrow_file), 'rb') as source:
            try:
                reader = ipc.open_stream(source)
            except:
                source.seek(0)
                reader = ipc.open_file(source)

            batches = []
            for batch in reader:
                batches.append(batch)

            if batches:
                table = pa.Table.from_batches(batches)
                data = table.to_pydict()

                # Add to combined dict
                for i, data_id in enumerate(data['dataID']):
                    all_data[data_id] = {
                        key: values[i] for key, values in data.items()
                    }

    print(f"  Total episodes loaded: {len(all_data)}")
    return all_data

def extract_all_turns(episode_data: Dict) -> List[Dict]:
    """Extract all turns from 5 sessions into a single list"""
    all_turns = []

    for session in ['first', 'second', 'third', 'fourth', 'fifth']:
        dialogue_key = f"{session}_session_dialogue"
        speakers_key = f"{session}_session_speakers"

        dialogue = episode_data.get(dialogue_key, [])
        speakers = episode_data.get(speakers_key, [])

        for utterance, speaker in zip(dialogue, speakers):
            all_turns.append({
                "session": session,
                "speaker": speaker,
                "utterance": utterance
            })

    return all_turns

def build_persona_turns(all_turns: List[Dict], target_speaker: str, context_window: int = 3) -> List[Dict]:
    """
    Extract turns where target speaker speaks, with conversation context

    Args:
        all_turns: All conversation turns
        target_speaker: Speaker to extract (e.g., "Husband")
        context_window: Number of previous turns to include as context

    Returns:
        List of turn dicts with context, input, ground_truth
    """
    persona_turns = []

    for i, turn in enumerate(all_turns):
        # Skip if not target speaker or first turn
        if turn["speaker"] != target_speaker or i == 0:
            continue

        # Build context from previous turns
        context_start = max(0, i - context_window)
        context_turns = all_turns[context_start:i]

        context = "\n".join([
            f"{t['speaker']}: {t['utterance']}"
            for t in context_turns
        ])

        # Input is the immediately previous utterance (from partner)
        input_utterance = all_turns[i - 1]["utterance"]

        # Ground truth is current speaker's utterance
        ground_truth = turn["utterance"]

        persona_turns.append({
            "turn_id": len(persona_turns) + 1,
            "session": turn["session"],
            "context": context,
            "input": input_utterance,
            "ground_truth": ground_truth
        })

    return persona_turns

def split_train_test(turns: List[Dict], train_ratio: float = 0.5) -> tuple:
    """Split turns into train and test sets"""
    split_idx = int(len(turns) * train_ratio)
    return turns[:split_idx], turns[split_idx:]

def process_persona(persona_id: str, episode_data: Dict, profile: Dict, output_dir: Path):
    """Process a single persona and save train/test turns"""
    print(f"\n{'='*80}")
    print(f"Processing: {persona_id}")
    print(f"{'='*80}")

    # Extract all turns from episode
    all_turns = extract_all_turns(episode_data)
    print(f"  Total turns in episode: {len(all_turns)}")

    # Get target speaker from profile
    # Handle cases like "Neighbors" + "_A" -> "Neighbors A"
    base_speaker = profile["speaker_role"]
    suffix = persona_id.split('_')[-1]  # Get "A" or "B"

    # Try exact match first, then with suffix
    target_speaker = base_speaker
    speakers_in_episode = set(turn["speaker"] for turn in all_turns)

    if base_speaker not in speakers_in_episode:
        # Try with suffix (e.g., "Neighbors" -> "Neighbors A")
        target_speaker = f"{base_speaker} {suffix}"
        if target_speaker not in speakers_in_episode:
            print(f"  ⚠️  WARNING: Neither '{base_speaker}' nor '{target_speaker}' found in episode")
            print(f"  Available speakers: {speakers_in_episode}")
            target_speaker = base_speaker  # Fall back to original

    print(f"  Target speaker: {target_speaker}")

    # Build persona turns with context
    persona_turns = build_persona_turns(all_turns, target_speaker, context_window=3)
    print(f"  Persona turns extracted: {len(persona_turns)}")

    if len(persona_turns) == 0:
        print(f"  ⚠️  WARNING: No turns found for {persona_id}")
        return

    # Split into train/test
    train_turns, test_turns = split_train_test(persona_turns, train_ratio=0.5)
    print(f"  Train turns: {len(train_turns)}")
    print(f"  Test turns: {len(test_turns)}")

    # Prepare output data
    train_data = {
        "persona_id": persona_id,
        "speaker_role": profile["speaker_role"],
        "relationship": profile["relationship"],
        "total_turns": len(train_turns),
        "turns": train_turns
    }

    test_data = {
        "persona_id": persona_id,
        "speaker_role": profile["speaker_role"],
        "relationship": profile["relationship"],
        "total_turns": len(test_turns),
        "turns": test_turns
    }

    # Save to files
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_turns.json"
    test_path = output_dir / "test_turns.json"

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Saved:")
    print(f"     {train_path}")
    print(f"     {test_path}")

    # Print example
    if train_turns:
        print(f"\n  Example turn:")
        example = train_turns[0]
        print(f"    Context: {example['context'][:100]}...")
        print(f"    Input: {example['input'][:80]}...")
        print(f"    Ground truth: {example['ground_truth'][:80]}...")

def main():
    # Paths
    arrow_dir = Path("data/raw/cc/conversationchronicles/train")
    personas_dir = Path("personas_cc")

    print("="*80)
    print("Conversation Chronicles ターン抽出")
    print("="*80)

    # Load all arrow data
    all_episodes = load_arrow_data(arrow_dir)

    # Get all persona directories
    persona_dirs = [d for d in personas_dir.iterdir() if d.is_dir()]
    print(f"\nFound {len(persona_dirs)} personas")

    # Process each persona
    success_count = 0
    for persona_dir in sorted(persona_dirs):
        persona_id = persona_dir.name

        # Load profile
        profile_path = persona_dir / "profile.json"
        if not profile_path.exists():
            print(f"⚠️  Skipping {persona_id}: No profile.json")
            continue

        with open(profile_path) as f:
            profile = json.load(f)

        # Extract episode ID (e.g., "episode-184019_A" -> "episode-184019")
        episode_id = persona_id.split('_')[0]

        if episode_id not in all_episodes:
            print(f"⚠️  Skipping {persona_id}: Episode {episode_id} not found in dataset")
            continue

        # Process this persona
        try:
            process_persona(
                persona_id=persona_id,
                episode_data=all_episodes[episode_id],
                profile=profile,
                output_dir=persona_dir
            )
            success_count += 1
        except Exception as e:
            print(f"❌ Error processing {persona_id}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total personas: {len(persona_dirs)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(persona_dirs) - success_count}")
    print("="*80)

if __name__ == "__main__":
    main()
