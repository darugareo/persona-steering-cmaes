#!/usr/bin/env python3
"""
Conversation Chronicles の Arrow データ構造を確認
pyarrow直接読み込み (streaming reader)
"""
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc

def inspect_arrow_file(arrow_path):
    """Inspect Arrow file structure using pyarrow streaming reader"""
    print(f"Reading: {arrow_path}")

    # Try to read as IPC stream
    with pa.OSFile(str(arrow_path), 'rb') as source:
        try:
            # Try stream reader first
            reader = ipc.open_stream(source)
        except:
            # Fall back to file reader
            source.seek(0)
            reader = ipc.open_file(source)

        # Get schema
        schema = reader.schema
        print("\n=== Schema ===")
        for field in schema:
            print(f"  {field.name}: {field.type}")

        # Read all batches
        batches = []
        try:
            for batch in reader:
                batches.append(batch)
        except StopIteration:
            pass

        if not batches:
            print("No batches found")
            return None

        # Combine into table
        table = pa.Table.from_batches(batches)
        print(f"\n=== Total rows: {len(table)} ===")

        # Convert to dict
        data = table.to_pydict()

    # Find episode-184019
    episode_idx = None
    for i, data_id in enumerate(data['dataID']):
        if 'episode-184019' in data_id:
            episode_idx = i
            break

    if episode_idx is None:
        print("\nepisode-184019 not found in this file")
        return None

    print(f"\n=== Found episode-184019 at index {episode_idx} ===")

    # Extract single example
    example = {key: values[episode_idx] for key, values in data.items()}

    # Print structure
    print("\n=== Example Keys ===")
    for key in example.keys():
        value = example[key]
        if isinstance(value, list):
            print(f"  {key}: list[{len(value)}]")
        else:
            print(f"  {key}: {type(value).__name__}")

    # Print session structure
    print("\n=== Session Structure ===")
    for session_name in ['first', 'second', 'third', 'fourth', 'fifth']:
        speakers_key = f"{session_name}_session_speakers"
        dialogue_key = f"{session_name}_session_dialogue"

        speakers = example.get(speakers_key, [])
        dialogue = example.get(dialogue_key, [])

        if speakers:
            print(f"\n{session_name.capitalize()} Session:")
            print(f"  Num turns: {len(speakers)}")
            print(f"  Speakers: {speakers[:5]}...")
            print(f"  First utterance: {dialogue[0] if dialogue else 'N/A'}")

    return example

def main():
    arrow_files = list(Path("data/raw/cc/conversationchronicles/train").glob("*.arrow"))

    print("="*80)
    print("Conversation Chronicles データ構造確認")
    print("="*80)

    for arrow_file in arrow_files:
        example = inspect_arrow_file(arrow_file)
        if example:
            # Save example
            output_path = Path("results/cc_example_structure.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(example, f, indent=2, ensure_ascii=False)

            print(f"\n✅ Saved example to: {output_path}")
            break

if __name__ == "__main__":
    main()
