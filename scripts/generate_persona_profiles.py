#!/usr/bin/env python3
"""
Generate Persona Profiles for 7 New Personas
==============================================

Purpose:
  Create persona profiles (style descriptions) for new personas
  to be used in CMA-ES optimization and judge evaluation.

Input:
  personas/{persona_id}/persona_samples.json

Output:
  persona_profiles/{persona_id}.json
  all_persona_profiles.json (10 personas total)
"""

import json
import os
from pathlib import Path
from openai import OpenAI

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment")

client = OpenAI(api_key=OPENAI_API_KEY)

# 7 new personas to process
NEW_PERSONAS = [
    "episode-5289_A",
    "episode-29600_A",
    "episode-88279_B",
    "episode-132247_A",
    "episode-166805_A",
    "episode-196697_B",
    "episode-225888_A"
]

# Existing 3 personas (load from existing files)
EXISTING_PERSONAS = [
    "episode-184019_A",
    "episode-239427_A",
    "episode-118328_B"
]

# Profile generation prompt
PROFILE_PROMPT_TEMPLATE = """You are a linguistic analyst. Your task is to analyze conversational utterances and describe the speaker's COMMUNICATION STYLE ONLY.

DO NOT:
- Describe personality traits or character
- Mention specific topics or content
- Make assumptions about demographics

DO:
- Describe formality level (formal/casual/neutral)
- Describe emotional expression (emotionally expressive/reserved/neutral)
- Describe verbosity (verbose/concise/balanced)
- Describe interpersonal distance (warm/detached/professional)
- Describe sentence structure patterns
- Note discourse markers, hedging, or directness

Below are {num_samples} conversational samples from the same speaker:

{samples}

Provide a concise profile (2-4 sentences) describing ONLY this speaker's communication style.
"""


def load_persona_samples(persona_id: str) -> dict:
    """Load persona samples from JSON file"""
    samples_path = Path(f"personas/{persona_id}/persona_samples.json")

    if not samples_path.exists():
        raise FileNotFoundError(f"Samples not found: {samples_path}")

    with open(samples_path) as f:
        return json.load(f)


def format_samples_for_prompt(samples_data: dict) -> str:
    """Format samples into readable text for LLM"""
    samples = samples_data.get("samples", [])

    formatted = []
    for i, sample in enumerate(samples, 1):
        text = sample.get("text", "").strip()
        relationship = sample.get("relationship", "Unknown")
        formatted.append(f"--- Sample {i} (Context: {relationship}) ---\n{text}\n")

    return "\n".join(formatted)


def generate_profile_with_llm(persona_id: str, samples_data: dict) -> str:
    """Generate persona profile using GPT-4o-mini"""

    samples = samples_data.get("samples", [])
    num_samples = len(samples)

    # Format samples for prompt
    samples_text = format_samples_for_prompt(samples_data)

    # Create prompt
    prompt = PROFILE_PROMPT_TEMPLATE.format(
        num_samples=num_samples,
        samples=samples_text
    )

    print(f"  Generating profile for {persona_id}...")
    print(f"    Using {num_samples} samples")

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a linguistic style analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=300
    )

    profile = response.choices[0].message.content.strip()

    print(f"    ✓ Profile generated ({len(profile)} chars)")

    return profile


def save_individual_profile(persona_id: str, profile: str):
    """Save individual persona profile to JSON"""
    output_dir = Path("persona_profiles")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{persona_id}.json"

    data = {
        "persona_id": persona_id,
        "profile": profile
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"    ✓ Saved to {output_path}")


def load_existing_profiles() -> dict:
    """Load existing 3 persona profiles"""
    profiles = {}

    for persona_id in EXISTING_PERSONAS:
        profile_path = Path(f"persona_profiles/{persona_id}.json")

        if not profile_path.exists():
            print(f"    ⚠️  Missing existing profile: {persona_id}")
            continue

        with open(profile_path) as f:
            data = json.load(f)
            profiles[persona_id] = data.get("profile", "")

    return profiles


def create_unified_profile_file(all_profiles: dict):
    """Create all_persona_profiles.json with 10 personas"""
    output_path = Path("all_persona_profiles.json")

    # Format: {"persona_id": {"profile": "..."}, ...}
    formatted = {
        pid: {"profile": profile}
        for pid, profile in all_profiles.items()
    }

    with open(output_path, 'w') as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Created unified profile: {output_path}")
    print(f"  Total personas: {len(formatted)}")


def verify_profiles(all_profiles: dict):
    """Verify all profiles are valid"""
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    issues = []

    for persona_id, profile in all_profiles.items():
        if not profile or len(profile.strip()) == 0:
            issues.append(f"  ✗ {persona_id}: Empty profile")
        elif len(profile) < 50:
            issues.append(f"  ⚠️  {persona_id}: Very short profile ({len(profile)} chars)")
        else:
            print(f"  ✓ {persona_id}: {len(profile)} chars")

    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(issue)
        return False

    print("\n✓ All profiles valid")
    return True


def main():
    print("=" * 80)
    print("PERSONA PROFILE GENERATION")
    print("=" * 80)

    # Step 1: Load existing profiles
    print("\n" + "-" * 80)
    print("Step 1: Loading existing profiles (3 personas)")
    print("-" * 80)

    all_profiles = load_existing_profiles()
    print(f"✓ Loaded {len(all_profiles)} existing profiles")

    # Step 2: Generate new profiles
    print("\n" + "-" * 80)
    print("Step 2: Generating new profiles (7 personas)")
    print("-" * 80)

    for persona_id in NEW_PERSONAS:
        print(f"\n{persona_id}:")

        try:
            # Load samples
            samples_data = load_persona_samples(persona_id)

            # Generate profile
            profile = generate_profile_with_llm(persona_id, samples_data)

            # Save individual file
            save_individual_profile(persona_id, profile)

            # Add to collection
            all_profiles[persona_id] = profile

        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue

    # Step 3: Create unified file
    print("\n" + "-" * 80)
    print("Step 3: Creating unified profile file")
    print("-" * 80)

    create_unified_profile_file(all_profiles)

    # Step 4: Verify
    verify_profiles(all_profiles)

    print("\n" + "=" * 80)
    print("✓ PROFILE GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  • persona_profiles/*.json (7 new files)")
    print(f"  • all_persona_profiles.json (10 personas)")
    print(f"\nNext step:")
    print(f"  python scripts/run_7personas_optimization.py")


if __name__ == "__main__":
    main()
