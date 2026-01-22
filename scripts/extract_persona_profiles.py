#!/usr/bin/env python3
"""
Extract persona profiles from ConversationChronicles for persona-aware evaluation.

This script creates detailed persona profiles that include:
- Trait values (R1-R5)
- Behavioral patterns
- Communication style
- Example responses from actual conversations
- Value priorities
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from typing import Dict, List, Any
from collections import Counter, defaultdict
import argparse
import pyarrow.parquet as pq


def load_persona_data(parquet_path: Path) -> List[Dict]:
    """Load persona session data from parquet file."""
    print(f"Loading data from: {parquet_path}")
    table = pq.read_table(parquet_path)
    data = table.to_pylist()
    print(f"Loaded {len(data)} sessions")

    # Count unique personas
    unique_personas = len(set(row['persona_id'] for row in data))
    print(f"Found {unique_personas} unique personas")
    return data


def extract_communication_style(texts: List[str]) -> Dict[str, Any]:
    """
    Analyze communication style from persona's utterances.

    Returns:
        Dictionary with style characteristics
    """
    if not texts:
        return {}

    # Combine all texts
    combined_text = " ".join(texts)

    # Simple heuristics for communication style
    avg_length = sum(len(t) for t in texts) / len(texts)

    # Count punctuation for emotional expressiveness
    exclamation_count = combined_text.count('!')
    question_count = combined_text.count('?')

    # Count first-person pronouns
    i_count = combined_text.lower().count(' i ')
    we_count = combined_text.lower().count(' we ')

    # Formality heuristics (very basic)
    informal_words = ['yeah', 'yep', 'nope', 'gonna', 'wanna', 'kinda', 'sorta']
    informal_count = sum(combined_text.lower().count(w) for w in informal_words)

    return {
        "avg_utterance_length": avg_length,
        "exclamation_rate": exclamation_count / len(texts) if texts else 0,
        "question_rate": question_count / len(texts) if texts else 0,
        "first_person_singular_rate": i_count / len(texts) if texts else 0,
        "first_person_plural_rate": we_count / len(texts) if texts else 0,
        "informal_language_rate": informal_count / len(texts) if texts else 0,
        "formality": "informal" if informal_count > 2 else "neutral",
    }


def extract_behavioral_patterns(persona_sessions: List[Dict]) -> Dict[str, Any]:
    """
    Extract behavioral patterns from persona's sessions.

    Args:
        persona_sessions: List of session dictionaries for single persona

    Returns:
        Dictionary describing behavioral patterns
    """
    # Get all texts (try both 'text' and 'session_text' fields)
    texts = []
    for row in persona_sessions:
        text = row.get('text') or row.get('session_text')
        if text and isinstance(text, str):
            texts.append(text)

    if not texts:
        return {"error": "No text data available"}

    # Communication style
    comm_style = extract_communication_style(texts)

    # Relationship contexts
    relationships = Counter(row.get('relationship', 'unknown') for row in persona_sessions)

    # Average traits across sessions
    trait_cols = ['directness', 'emotional_valence', 'social_orientation', 'audience_focus']
    trait_means = {}
    for col in trait_cols:
        values = [row[col] for row in persona_sessions if col in row and row[col] is not None]
        if values:
            trait_means[col] = sum(values) / len(values)

    return {
        "communication_style": comm_style,
        "relationship_contexts": dict(relationships),
        "trait_averages": trait_means,
        "num_sessions": len(persona_sessions),
        "total_utterances": len(texts),
    }


def get_example_responses(persona_sessions: List[Dict], n_examples: int = 10) -> List[Dict]:
    """
    Get representative example responses from persona's conversations.

    Args:
        persona_sessions: List of session dictionaries for single persona
        n_examples: Number of examples to extract

    Returns:
        List of example response dictionaries
    """
    examples = []

    # Sample diverse examples
    import random
    random.seed(42)
    sampled = random.sample(persona_sessions, min(n_examples, len(persona_sessions)))

    for row in sampled:
        text = row.get('text') or row.get('session_text', '')
        example = {
            "text": text,
            "relationship": row.get('relationship', 'unknown'),
            "session_idx": row.get('session_idx', -1),
        }

        # Add trait values if available
        trait_cols = ['directness', 'emotional_valence', 'social_orientation', 'audience_focus']
        example["traits"] = {}
        for col in trait_cols:
            if col in row and row[col] is not None:
                example["traits"][col] = float(row[col])

        examples.append(example)

    return examples


def infer_values_and_priorities(behavioral_patterns: Dict, trait_averages: Dict) -> Dict[str, Any]:
    """
    Infer value priorities from behavioral patterns and traits.

    Args:
        behavioral_patterns: Behavioral pattern dictionary
        trait_averages: Average trait values

    Returns:
        Dictionary describing inferred values
    """
    priorities = []

    # Based on social_orientation
    if 'social_orientation' in trait_averages:
        if trait_averages['social_orientation'] > 0.5:
            priorities.append("values social harmony and cooperation")
        elif trait_averages['social_orientation'] < -0.5:
            priorities.append("values independence and self-reliance")

    # Based on emotional_valence
    if 'emotional_valence' in trait_averages:
        if trait_averages['emotional_valence'] > 0.5:
            priorities.append("expresses emotions positively")
        elif trait_averages['emotional_valence'] < -0.5:
            priorities.append("tends toward critical or negative expression")

    # Based on directness
    if 'directness' in trait_averages:
        if trait_averages['directness'] > 0.7:
            priorities.append("values directness and clarity")
        elif trait_averages['directness'] < 0.3:
            priorities.append("prefers indirect or nuanced communication")

    # Based on communication style
    comm_style = behavioral_patterns.get('communication_style', {})
    if comm_style.get('first_person_plural_rate', 0) > comm_style.get('first_person_singular_rate', 0):
        priorities.append("thinks in terms of 'we' rather than 'I'")

    return {
        "inferred_priorities": priorities,
        "confidence": "low" if len(priorities) < 2 else "medium",
    }


def create_persona_profile(persona_id: str, all_data: List[Dict]) -> Dict[str, Any]:
    """
    Create comprehensive persona profile for a single persona.

    Args:
        persona_id: Persona identifier
        all_data: Full dataset as list of dictionaries

    Returns:
        Complete persona profile dictionary
    """
    # Filter to this persona
    persona_sessions = [row for row in all_data if row.get('persona_id') == persona_id]

    if len(persona_sessions) == 0:
        return {"error": f"No data found for persona {persona_id}"}

    # Extract components
    behavioral_patterns = extract_behavioral_patterns(persona_sessions)
    example_responses = get_example_responses(persona_sessions, n_examples=10)

    # Infer values
    values = infer_values_and_priorities(
        behavioral_patterns,
        behavioral_patterns.get('trait_averages', {})
    )

    # Compile profile
    profile = {
        "persona_id": persona_id,
        "num_sessions": len(persona_sessions),
        "trait_averages": behavioral_patterns.get('trait_averages', {}),
        "communication_style": behavioral_patterns.get('communication_style', {}),
        "relationship_contexts": behavioral_patterns.get('relationship_contexts', {}),
        "values_and_priorities": values,
        "example_responses": example_responses,
        "metadata": {
            "source": "ConversationChronicles",
            "extraction_date": "2025-12-08",
        }
    }

    return profile


def extract_all_personas(
    parquet_path: Path,
    output_dir: Path,
    persona_ids: List[str] = None,
    max_personas: int = None
):
    """
    Extract profiles for all (or specified) personas.

    Args:
        parquet_path: Path to persona session data
        output_dir: Directory to save profiles
        persona_ids: Optional list of specific personas to extract
        max_personas: Optional limit on number of personas
    """
    # Load data
    all_data = load_persona_data(parquet_path)

    # Determine which personas to process
    if persona_ids:
        personas_to_process = persona_ids
    else:
        all_persona_ids = list(set(row['persona_id'] for row in all_data))
        if max_personas:
            personas_to_process = all_persona_ids[:max_personas]
        else:
            personas_to_process = all_persona_ids

    print(f"\nProcessing {len(personas_to_process)} personas...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract profiles
    profiles = {}
    for persona_id in personas_to_process:
        print(f"  Extracting: {persona_id}")
        profile = create_persona_profile(persona_id, all_data)
        profiles[persona_id] = profile

        # Save individual profile
        profile_path = output_dir / f"{persona_id}.json"
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)

    # Save combined profiles
    combined_path = output_dir / "all_persona_profiles.json"
    with open(combined_path, 'w') as f:
        json.dump(profiles, f, indent=2)

    # Save summary
    summary = {
        "num_personas": len(profiles),
        "persona_ids": list(profiles.keys()),
        "extraction_date": "2025-12-08",
        "source_file": str(parquet_path),
    }
    summary_path = output_dir / "profiles_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Extracted {len(profiles)} persona profiles")
    print(f"  Individual profiles: {output_dir}/<persona_id>.json")
    print(f"  Combined file: {combined_path}")
    print(f"  Summary: {summary_path}")

    return profiles


def main():
    parser = argparse.ArgumentParser(
        description="Extract persona profiles from ConversationChronicles"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/cc/filtered/persona_session_docs_filtered.parquet"),
        help="Input parquet file with persona sessions"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/persona_profiles"),
        help="Output directory for persona profiles"
    )
    parser.add_argument(
        "--persona-ids",
        nargs="+",
        help="Specific persona IDs to extract (optional)"
    )
    parser.add_argument(
        "--max-personas",
        type=int,
        help="Maximum number of personas to process (optional)"
    )

    args = parser.parse_args()

    # Extract profiles
    profiles = extract_all_personas(
        parquet_path=args.input,
        output_dir=args.output_dir,
        persona_ids=args.persona_ids,
        max_personas=args.max_personas
    )

    # Print sample profile
    if profiles:
        sample_id = list(profiles.keys())[0]
        print(f"\n{'='*60}")
        print(f"Sample Profile: {sample_id}")
        print(f"{'='*60}")
        print(json.dumps(profiles[sample_id], indent=2)[:500] + "...")


if __name__ == "__main__":
    main()
