"""
Extract representative prompts from filtered persona dataset
Select diverse examples covering different trait characteristics
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import numpy as np


def score_text_for_traits(text: str) -> dict:
    """Score text for trait signals"""
    if not isinstance(text, str):
        return {trait: 0.0 for trait in ['directness', 'emotional_valence', 'social_orientation', 'audience_focus']}

    t = text.lower()
    tokens = re.findall(r'[a-z]+', t)
    n = max(len(tokens), 1)
    token_counts = Counter(tokens)

    hedge_words = {'maybe', 'might', 'could', 'perhaps', 'suggest', 'consider', 'possibly', 'probably'}
    pos_words = {'good', 'great', 'happy', 'glad', 'excited', 'wonderful', 'excellent', 'amazing'}
    neg_words = {'bad', 'sad', 'angry', 'upset', 'worried', 'stressed', 'frustrated', 'terrible'}
    we_words = {'we', 'together', 'team', 'our', 'us'}
    you_words = {'you', 'your', 'yours'}

    scores = {
        'directness': 1.0 - sum(token_counts[w] for w in hedge_words) / n,
        'emotional_valence': (sum(token_counts[w] for w in pos_words) - sum(token_counts[w] for w in neg_words)) / n,
        'social_orientation': sum(token_counts[w] for w in we_words) / n,
        'audience_focus': sum(token_counts[w] for w in you_words) / n,
    }
    return scores


def extract_diverse_prompts(
    session_df: pd.DataFrame,
    n_prompts: int = 20,
    min_length: int = 50,
    max_length: int = 500
) -> list:
    """
    Extract diverse prompts covering trait space

    Args:
        session_df: Filtered session DataFrame
        n_prompts: Number of prompts to extract
        min_length: Minimum text length
        max_length: Maximum text length

    Returns:
        List of prompt dictionaries
    """
    print(f"Extracting {n_prompts} diverse prompts...")

    # Filter by length
    session_df = session_df.copy()
    session_df['text_len'] = session_df['session_text'].str.len()
    session_df = session_df[
        (session_df['text_len'] >= min_length) &
        (session_df['text_len'] <= max_length)
    ]

    print(f"After length filter: {len(session_df)} sessions")

    # Score all sessions
    print("Scoring sessions for trait signals...")
    trait_scores = []
    for idx, row in session_df.iterrows():
        scores = score_text_for_traits(row['session_text'])
        trait_scores.append({
            'persona_id': row['persona_id'],
            'session_idx': row['session_idx'],
            'relationship': row['relationship'],
            'text': row['session_text'][:200],  # Preview
            **scores
        })

    trait_df = pd.DataFrame(trait_scores)

    # Bin traits for stratified sampling
    trait_cols = ['directness', 'emotional_valence', 'social_orientation', 'audience_focus']
    for col in trait_cols:
        try:
            trait_df[f'{col}_bin'] = pd.qcut(trait_df[col], q=4, labels=['low', 'med-low', 'med-high', 'high'], duplicates='drop')
        except ValueError:
            # Fallback to simple binning if qcut fails
            trait_df[f'{col}_bin'] = pd.cut(trait_df[col], bins=4, labels=['low', 'med-low', 'med-high', 'high'])

    # Sample diverse prompts
    selected = []

    # Strategy: sample from different trait combinations
    # Priority 1: Extreme values (high/low for each trait)
    for trait in trait_cols:
        # High extreme
        high_subset = trait_df[trait_df[f'{trait}_bin'] == 'high'].sample(min(2, len(trait_df[trait_df[f'{trait}_bin'] == 'high'])), random_state=42)
        selected.extend(high_subset.to_dict('records'))

        # Low extreme
        low_subset = trait_df[trait_df[f'{trait}_bin'] == 'low'].sample(min(2, len(trait_df[trait_df[f'{trait}_bin'] == 'low'])), random_state=42)
        selected.extend(low_subset.to_dict('records'))

    # Priority 2: Fill remaining slots with diverse samples
    remaining_df = trait_df[~trait_df['persona_id'].isin([s['persona_id'] for s in selected])]
    if len(selected) < n_prompts and len(remaining_df) > 0:
        n_fill = min(n_prompts - len(selected), len(remaining_df))
        fill_samples = remaining_df.sample(n_fill, random_state=42)
        selected.extend(fill_samples.to_dict('records'))

    # Format as prompt set
    prompts = []
    for i, item in enumerate(selected[:n_prompts]):
        # Get full text from original dataframe
        full_text = session_df[
            (session_df['persona_id'] == item['persona_id']) &
            (session_df['session_idx'] == item['session_idx'])
        ]['session_text'].iloc[0]

        prompts.append({
            'id': f'prompt_{i+1:02d}',
            'persona_id': item['persona_id'],
            'session_idx': int(item['session_idx']),
            'relationship': item['relationship'],
            'text': full_text,
            'traits': {
                'directness': float(item['directness']),
                'emotional_valence': float(item['emotional_valence']),
                'social_orientation': float(item['social_orientation']),
                'audience_focus': float(item['audience_focus'])
            },
            'length': len(full_text)
        })

    return prompts


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract prompts from filtered personas")
    parser.add_argument('--data_dir', type=str, default='data/processed/cc/filtered',
                       help='Directory with filtered data')
    parser.add_argument('--output', type=str, default='data/prompts/prompts_v2.json',
                       help='Output JSON file')
    parser.add_argument('--n_prompts', type=int, default=20,
                       help='Number of prompts to extract')
    parser.add_argument('--min_length', type=int, default=50,
                       help='Minimum text length')
    parser.add_argument('--max_length', type=int, default=500,
                       help='Maximum text length')

    args = parser.parse_args()

    # Load filtered data
    print("="*60)
    print("Prompt Extraction from Filtered Personas")
    print("="*60)

    data_dir = Path(args.data_dir)
    session_df = pd.read_parquet(data_dir / 'persona_session_docs_filtered.parquet')

    print(f"\nLoaded {len(session_df)} sessions from {session_df['persona_id'].nunique()} personas")

    # Extract prompts
    prompts = extract_diverse_prompts(
        session_df,
        n_prompts=args.n_prompts,
        min_length=args.min_length,
        max_length=args.max_length
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_set = {
        'version': '2.0',
        'description': 'Prompts extracted from filtered ConversationChronicles personas',
        'n_prompts': len(prompts),
        'source': 'persona_session_docs_filtered.parquet',
        'prompts': prompts
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_set, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Extracted {len(prompts)} prompts")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}")

    # Print summary
    print("\nTrait distribution:")
    trait_cols = ['directness', 'emotional_valence', 'social_orientation', 'audience_focus']
    for trait in trait_cols:
        values = [p['traits'][trait] for p in prompts]
        print(f"  {trait}: mean={np.mean(values):.3f}, std={np.std(values):.3f}, range=[{np.min(values):.3f}, {np.max(values):.3f}]")

    print("\nRelationship distribution:")
    rel_counts = Counter([p['relationship'] for p in prompts])
    for rel, count in rel_counts.most_common():
        print(f"  {rel}: {count}")

    print("\nLength statistics:")
    lengths = [p['length'] for p in prompts]
    print(f"  mean={np.mean(lengths):.0f}, std={np.std(lengths):.0f}, range=[{np.min(lengths)}, {np.max(lengths)}]")


if __name__ == "__main__":
    main()
