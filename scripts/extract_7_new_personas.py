#!/usr/bin/env python3
"""
Extract 7 new personas from Conversation Chronicles
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from datasets import load_dataset
from typing import Dict, List
from collections import defaultdict

# 7 new personas to extract
# Note: episode-31102 and episode-229805 not found in dataset
# Replaced with nearby episodes from same numeric range
NEW_PERSONAS = [
    "episode-5289_A",      # P4: Formal Emotional (Husband, early range)
    "episode-29600_A",     # P5: Casual Neutral (Parent, ~30k range, replaces 31102)
    "episode-88279_B",     # P6: Verbose Intimate (Classmates B, mid range)
    "episode-132247_A",    # P7: Concise Detached (Neighbors A, ~130k range)
    "episode-166805_A",    # P8: Conflict Avoidant (Neighbors A, ~167k range)
    "episode-196697_B",    # P9: Advice Centric (Classmates B, ~197k range)
    "episode-225888_A",    # P10: Reflective Philosophical (Classmates A, ~226k, replaces 229805)
]

def get_speaker_names_for_relationship(relationship: str, speaker_letter: str):
    """Map relationship type to speaker names."""
    # Relationship-specific speaker names
    if 'Classmates' in relationship:
        return f"Classmates {speaker_letter}"
    elif 'Husband' in relationship or 'Wife' in relationship:
        return 'Husband' if speaker_letter == 'A' else 'Wife'
    elif 'Neighbors' in relationship:
        return f"Neighbors {speaker_letter}"
    elif 'Friends' in relationship:
        return f"Friends {speaker_letter}"
    elif 'Colleagues' in relationship:
        return f"Colleagues {speaker_letter}"
    elif 'Parent' in relationship or 'Child' in relationship:
        return 'Parent' if speaker_letter == 'A' else 'Child'
    elif 'Siblings' in relationship:
        return f"Siblings {speaker_letter}"
    else:
        # Default: try both formats
        return [f"{relationship} {speaker_letter}", f"{relationship.rstrip('s')} {speaker_letter}"]


def extract_persona_conversations(persona_id: str, dataset) -> List[Dict]:
    """Extract all conversations for a persona"""
    print(f"\n{'='*60}")
    print(f"Extracting: {persona_id}")
    print(f"{'='*60}")

    # Parse persona_id: episode-5289_A -> episode=5289, speaker=A or B
    parts = persona_id.split('-')
    episode_num = parts[1].split('_')[0]
    speaker_letter = parts[1].split('_')[1]  # A or B

    data_id = f"episode-{episode_num}"

    conversations = []

    # Search in train split
    for example in dataset['train']:
        if example['dataID'] == data_id:
            print(f"  Found episode: {data_id}")
            relationship = example.get('relationship', 'Unknown')
            print(f"  Relationship: {relationship}")

            # Get appropriate speaker names
            speaker_names = get_speaker_names_for_relationship(relationship, speaker_letter)
            if isinstance(speaker_names, str):
                speaker_names = [speaker_names]

            print(f"  Looking for speakers: {speaker_names}")

            # Extract conversations for each session
            session_names = ['first', 'second', 'third', 'fourth', 'fifth']

            for i, session_name in enumerate(session_names, start=1):
                dialogue_key = f'{session_name}_session_dialogue'
                speakers_key = f'{session_name}_session_speakers'

                if dialogue_key in example and speakers_key in example:
                    dialogues = example[dialogue_key]
                    speakers = example[speakers_key]

                    # Extract utterances for this speaker
                    speaker_utts = []
                    for dialogue, speaker in zip(dialogues, speakers):
                        if speaker in speaker_names:
                            speaker_utts.append(dialogue)

                    if speaker_utts:
                        conversations.append({
                            'session': i,
                            'relationship': relationship,
                            'utterances': speaker_utts,
                            'full_text': '\n'.join(speaker_utts)
                        })

            break  # Found the episode, no need to continue

    print(f"  Found {len(conversations)} sessions")
    total_utts = sum(len(c['utterances']) for c in conversations)
    print(f"  Total utterances: {total_utts}")

    return conversations


def compute_communication_stats(conversations: List[Dict]) -> Dict:
    """Compute basic communication statistics"""
    all_text = ' '.join(c['full_text'] for c in conversations)
    all_utts = []
    for c in conversations:
        all_utts.extend(c['utterances'])

    if not all_utts:
        return {}

    avg_length = sum(len(u) for u in all_utts) / len(all_utts)
    exclamation_rate = all_text.count('!') / len(all_utts)
    question_rate = all_text.count('?') / len(all_utts)

    # Pronouns
    i_count = all_text.lower().count(' i ')
    we_count = all_text.lower().count(' we ')
    you_count = all_text.lower().count(' you ')

    return {
        'avg_utterance_length': avg_length,
        'exclamation_rate': exclamation_rate,
        'question_rate': question_rate,
        'first_person_singular_rate': i_count / len(all_utts),
        'first_person_plural_rate': we_count / len(all_utts),
        'second_person_rate': you_count / len(all_utts),
        'total_utterances': len(all_utts),
        'num_sessions': len(conversations)
    }


def save_persona_profile(persona_id: str, conversations: List[Dict], output_dir: Path):
    """Save persona profile to directory"""
    persona_dir = output_dir / persona_id
    persona_dir.mkdir(exist_ok=True, parents=True)

    # Compute stats
    stats = compute_communication_stats(conversations)

    # Create profile
    profile = {
        'persona_id': persona_id,
        'num_sessions': len(conversations),
        'communication_style': stats,
        'relationship_contexts': {c['relationship'] for c in conversations},
        'example_responses': [
            {
                'text': c['full_text'][:500],  # First 500 chars
                'relationship': c['relationship'],
                'session_idx': c['session']
            }
            for c in conversations[:3]  # First 3 examples
        ],
        'metadata': {
            'source': 'ConversationChronicles',
            'extraction_date': '2025-12-15'
        }
    }

    # Save as JSON
    with open(persona_dir / 'persona_profile.json', 'w') as f:
        json.dump(profile, f, indent=2, default=str)

    # Save samples
    samples = {
        'persona_id': persona_id,
        'samples': [
            {
                'session': c['session'],
                'text': c['full_text'],
                'relationship': c['relationship']
            }
            for c in conversations
        ]
    }

    with open(persona_dir / 'persona_samples.json', 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"  ✓ Saved to: {persona_dir}")
    return profile


def main():
    print("="*80)
    print("EXTRACTING 7 NEW PERSONAS FROM CONVERSATION CHRONICLES")
    print("="*80)

    # Load dataset
    print("\nLoading Conversation Chronicles dataset...")
    dataset_path = "/data01/nakata/master_thesis/persona2/data/raw/cc/conversationchronicles"

    try:
        # Load only train split to avoid mixed format issues
        from datasets import Dataset
        dataset = {'train': Dataset.from_file(f"{dataset_path}/train/data-00000-of-00003.arrow")}
        print(f"✓ Loaded dataset train split (shard 1/3)")

        # Also load other shards
        dataset_shard2 = Dataset.from_file(f"{dataset_path}/train/data-00001-of-00003.arrow")
        dataset_shard3 = Dataset.from_file(f"{dataset_path}/train/data-00002-of-00003.arrow")

        # Concatenate all shards
        from datasets import concatenate_datasets
        dataset['train'] = concatenate_datasets([dataset['train'], dataset_shard2, dataset_shard3])
        print(f"✓ Concatenated all 3 shards: {len(dataset['train'])} examples")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # Output directory
    output_dir = Path("/data01/nakata/master_thesis/persona2/personas")
    output_dir.mkdir(exist_ok=True)

    # Extract each persona
    all_profiles = {}

    for persona_id in NEW_PERSONAS:
        conversations = extract_persona_conversations(persona_id, dataset)

        if conversations:
            profile = save_persona_profile(persona_id, conversations, output_dir)
            all_profiles[persona_id] = profile
        else:
            print(f"  ✗ No data found for {persona_id}")

    # Save summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for persona_id, profile in all_profiles.items():
        stats = profile['communication_style']
        print(f"\n{persona_id}:")
        print(f"  Sessions: {profile['num_sessions']}")
        print(f"  Utterances: {stats.get('total_utterances', 0)}")
        print(f"  Avg length: {stats.get('avg_utterance_length', 0):.1f}")
        print(f"  Exclamation rate: {stats.get('exclamation_rate', 0):.2f}")

    print(f"\n✓ Extracted {len(all_profiles)}/{len(NEW_PERSONAS)} personas")
    print(f"✓ Saved to: {output_dir}")


if __name__ == "__main__":
    main()
