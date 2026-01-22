#!/usr/bin/env python3
# main_generate_judge.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

from persona_judge import (
    load_conversations,
    extract_persona_features,
    select_representative_samples,
    generate_persona_profile,
    build_judge_prompt,
)


def generate_assets(base_dir: str | Path, persona_id: str) -> None:
    base_path = Path(base_dir)
    persona_dir = base_path / persona_id
    persona_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading conversations for persona {persona_id} from {persona_dir}")
    conversations = load_conversations(base_path, persona_id)

    print("Extracting persona features")
    features = extract_persona_features(conversations)

    print("Selecting representative samples")
    samples = select_representative_samples(conversations, n=10, max_len=300)

    print("Generating persona profile text")
    profile_text = generate_persona_profile(features, samples)

    print("Building judge prompt template")
    judge_prompt_template = build_judge_prompt(profile_text, samples)

    profile_path = persona_dir / "persona_profile.txt"
    samples_path = persona_dir / "persona_samples.json"
    judge_prompt_path = persona_dir / "final_judge_prompt.txt"
    features_path = persona_dir / "persona_features.json"

    print(f"Writing persona_profile to {profile_path}")
    profile_path.write_text(profile_text, encoding="utf8")

    print(f"Writing samples to {samples_path}")
    samples_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf8")

    print(f"Writing features to {features_path}")
    features_path.write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf8")

    print(f"Writing judge prompt template to {judge_prompt_path}")
    judge_prompt_path.write_text(judge_prompt_template, encoding="utf8")

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate persona-aware judge assets for a given persona id."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory where persona folders are stored.",
    )
    parser.add_argument(
        "--persona_id",
        type=str,
        required=True,
        help="Persona id, used as folder name under base_dir.",
    )
    args = parser.parse_args()

    generate_assets(args.base_dir, args.persona_id)


if __name__ == "__main__":
    main()
