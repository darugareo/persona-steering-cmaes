#!/usr/bin/env python3
"""
Convert existing persona profiles to raw_conversations.json format.

Reads from data/persona_profiles/*.json and creates
personas/{persona_id}/raw_conversations.json
"""

import json
from pathlib import Path
import argparse


def convert_persona_profile_to_conversations(persona_profile_path: Path, output_dir: Path):
    """
    Convert persona profile example_responses to conversation format.
    """
    with open(persona_profile_path) as f:
        profile = json.load(f)

    persona_id = profile["persona_id"]
    example_responses = profile.get("example_responses", [])

    conversations = []
    for example in example_responses:
        text = example.get("text", "")
        # Split by newlines to simulate turn-taking
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        # Alternate between user and assistant
        for i, line in enumerate(lines):
            role = "user" if i % 2 == 0 else "assistant"
            conversations.append({
                "role": role,
                "content": line
            })

    # Create output directory
    persona_dir = output_dir / persona_id
    persona_dir.mkdir(parents=True, exist_ok=True)

    # Write raw_conversations.json
    output_path = persona_dir / "raw_conversations.json"
    with open(output_path, "w", encoding="utf8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    print(f"Created {output_path}")
    print(f"  Total conversations: {len(conversations)}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--persona-profiles-dir",
        type=Path,
        default=Path("data/persona_profiles"),
        help="Directory containing persona profile JSONs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("personas"),
        help="Output directory for persona folders"
    )
    parser.add_argument(
        "--persona-id",
        type=str,
        default=None,
        help="Specific persona ID to convert (default: all)"
    )

    args = parser.parse_args()

    profiles_dir = args.persona_profiles_dir

    if args.persona_id:
        # Convert specific persona
        profile_path = profiles_dir / f"{args.persona_id}.json"
        if not profile_path.exists():
            print(f"Error: Profile not found: {profile_path}")
            return
        convert_persona_profile_to_conversations(profile_path, args.output_dir)
    else:
        # Convert all personas
        for profile_path in profiles_dir.glob("*.json"):
            if profile_path.name == "all_persona_profiles.json":
                continue
            if profile_path.name == "profiles_summary.json":
                continue

            print(f"\nConverting {profile_path.name}...")
            convert_persona_profile_to_conversations(profile_path, args.output_dir)


if __name__ == "__main__":
    main()
