#!/usr/bin/env python3
"""
Persona Selection Pipeline for IEEE Access Paper
=================================================

Goal: Select 10 diverse personas from Conversation Chronicles
Criteria:
  1. Minimum 40 utterances
  2. Diverse style profiles (formality, emotionality, verbosity, interpersonal)
  3. Maximally distant from existing 3 personas
  4. Cover 7 target archetypes
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import re

# Existing 3 personas (anchors)
EXISTING_PERSONAS = {
    "episode-184019_A": "formal/verbose/detached",
    "episode-239427_A": "casual/emotional/intimate",
    "episode-118328_B": "neutral/concise"
}

# Target 7 archetypes to fill
TARGET_ARCHETYPES = [
    "formal_emotional",      # P4
    "casual_neutral",        # P5
    "verbose_intimate",      # P6
    "concise_detached",      # P7
    "conflict_avoidant",     # P8
    "advice_centric",        # P9
    "reflective_philosophical" # P10
]


class StyleScorer:
    """Compute 4-dimensional style scores for personas"""

    # Emotion words (simplified)
    EMOTION_WORDS = {
        'love', 'hate', 'happy', 'sad', 'angry', 'excited', 'nervous',
        'worried', 'glad', 'grateful', 'sorry', 'afraid', 'frustrated',
        'disappointed', 'anxious', 'upset', 'hurt', 'confused', 'stressed'
    }

    # Formal markers
    FORMAL_MARKERS = {'however', 'therefore', 'furthermore', 'consequently', 'nevertheless'}

    # Interpersonal markers
    INTERPERSONAL_MARKERS = {'you', 'your', 'we', 'us', 'our', 'together'}

    def __init__(self):
        pass

    def compute_formality(self, text: str) -> float:
        """
        Formality score based on:
        - Average sentence length
        - Formal connectives
        - Punctuation complexity
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.5

        avg_sent_len = np.mean([len(s.split()) for s in sentences])
        formal_count = sum(1 for word in text.lower().split() if word in self.FORMAL_MARKERS)

        # Normalize
        formality = min(1.0, (avg_sent_len / 30.0) * 0.7 + (formal_count / len(sentences)) * 0.3)
        return formality

    def compute_emotionality(self, text: str, exclamation_rate: float) -> float:
        """
        Emotionality score based on:
        - Emotion words
        - Exclamation marks
        - Evaluative language
        """
        words = text.lower().split()
        emotion_count = sum(1 for w in words if w in self.EMOTION_WORDS)

        if not words:
            return 0.0

        emotion_density = emotion_count / len(words)
        emotionality = min(1.0, emotion_density * 20.0 * 0.6 + exclamation_rate * 0.4)
        return emotionality

    def compute_verbosity(self, avg_length: float) -> float:
        """Verbosity score from average utterance length"""
        # Normalize around typical range (200-1000 chars)
        return min(1.0, avg_length / 800.0)

    def compute_interpersonal(self, text: str, first_person_rate: float) -> float:
        """
        Interpersonal orientation based on:
        - Second person pronouns (you/your)
        - First person plural (we/us)
        - Empathy markers
        """
        words = text.lower().split()
        if not words:
            return 0.5

        interpersonal_count = sum(1 for w in words if w in self.INTERPERSONAL_MARKERS)
        interpersonal_density = interpersonal_count / len(words)

        # Combine with first-person usage
        interpersonal = min(1.0, interpersonal_density * 10.0 * 0.7 + first_person_rate * 0.03 * 0.3)
        return interpersonal

    def score_persona(self, profile: Dict) -> Dict[str, float]:
        """Compute 4D style vector for a persona"""
        style = profile.get('communication_style', {})
        examples = profile.get('example_responses', [])

        # Combine all example text
        all_text = ' '.join([ex.get('text', '') for ex in examples])

        scores = {
            'formality': self.compute_formality(all_text),
            'emotionality': self.compute_emotionality(
                all_text,
                style.get('exclamation_rate', 0.0)
            ),
            'verbosity': self.compute_verbosity(
                style.get('avg_utterance_length', 400.0)
            ),
            'interpersonal': self.compute_interpersonal(
                all_text,
                style.get('first_person_singular_rate', 5.0)
            )
        }

        return scores


def load_persona_profiles(path: str) -> Dict:
    """Load all persona profiles"""
    with open(path, 'r') as f:
        return json.load(f)


def filter_candidates(profiles: Dict, min_utterances: int = 40) -> List[str]:
    """STEP 1: Filter personas meeting minimum criteria"""
    candidates = []

    for persona_id, profile in profiles.items():
        # Check number of sessions (proxy for utterance count)
        num_sessions = profile.get('num_sessions', 0)

        # Check example responses exist
        examples = profile.get('example_responses', [])

        # Rough utterance count: num_sessions * ~10 utterances/session
        estimated_utterances = num_sessions * 10

        if estimated_utterances >= min_utterances and len(examples) >= 3:
            candidates.append(persona_id)

    return candidates


def compute_distance_to_anchors(
    candidate_scores: Dict[str, float],
    anchor_scores: Dict[str, Dict[str, float]]
) -> float:
    """Compute minimum distance to existing 3 personas"""
    candidate_vec = np.array([
        candidate_scores['formality'],
        candidate_scores['emotionality'],
        candidate_scores['verbosity'],
        candidate_scores['interpersonal']
    ])

    min_dist = float('inf')
    for anchor_id, anchor_score in anchor_scores.items():
        anchor_vec = np.array([
            anchor_score['formality'],
            anchor_score['emotionality'],
            anchor_score['verbosity'],
            anchor_score['interpersonal']
        ])
        dist = np.linalg.norm(candidate_vec - anchor_vec)
        min_dist = min(min_dist, dist)

    return min_dist


def assign_archetype(scores: Dict[str, float], examples: List[Dict]) -> str:
    """Heuristically assign persona to one of 7 target archetypes"""
    f = scores['formality']
    e = scores['emotionality']
    v = scores['verbosity']
    i = scores['interpersonal']

    # Check example text for specific patterns
    all_text = ' '.join([ex.get('text', '') for ex in examples]).lower()

    has_advice = any(word in all_text for word in ['should', 'recommend', 'suggest', 'advice', 'try'])
    has_conflict_avoid = any(phrase in all_text for phrase in ['i understand', 'i see', 'maybe', 'perhaps'])
    has_philosophical = any(word in all_text for word in ['meaning', 'purpose', 'believe', 'think about', 'reflect'])

    # Rule-based assignment
    if f > 0.6 and e > 0.5:
        return "formal_emotional"
    elif f < 0.4 and e < 0.4:
        return "casual_neutral"
    elif v > 0.7 and i > 0.6:
        return "verbose_intimate"
    elif v < 0.4 and i < 0.4:
        return "concise_detached"
    elif has_conflict_avoid and i > 0.5:
        return "conflict_avoidant"
    elif has_advice:
        return "advice_centric"
    elif has_philosophical or (f > 0.5 and v > 0.5):
        return "reflective_philosophical"
    else:
        # Default: assign based on dominant trait
        if max(f, e, v, i) == f:
            return "formal_emotional" if e > 0.4 else "reflective_philosophical"
        elif max(f, e, v, i) == e:
            return "conflict_avoidant"
        elif max(f, e, v, i) == v:
            return "verbose_intimate"
        else:
            return "advice_centric"


def main():
    print("=" * 80)
    print("PERSONA SELECTION PIPELINE")
    print("=" * 80)

    # Load data
    data_dir = Path("/data01/nakata/master_thesis/persona2/data/persona_profiles")
    profiles = load_persona_profiles(data_dir / "all_persona_profiles.json")

    print(f"\n✓ Loaded {len(profiles)} total personas from Chronicles")

    # STEP 1: Filter candidates
    print("\n" + "-" * 80)
    print("STEP 1: Filtering candidates (min 40 utterances)")
    print("-" * 80)

    candidates = filter_candidates(profiles, min_utterances=40)
    print(f"✓ {len(candidates)} candidates meet minimum criteria")

    # STEP 2: Compute style scores
    print("\n" + "-" * 80)
    print("STEP 2: Computing 4D style scores")
    print("-" * 80)

    scorer = StyleScorer()
    candidate_scores = {}

    for persona_id in candidates:
        profile = profiles[persona_id]
        scores = scorer.score_persona(profile)
        candidate_scores[persona_id] = scores

    print(f"✓ Computed scores for {len(candidate_scores)} candidates")

    # STEP 3: Compute distances to existing 3 personas
    print("\n" + "-" * 80)
    print("STEP 3: Computing distances to existing anchors")
    print("-" * 80)

    anchor_scores = {}
    for anchor_id in EXISTING_PERSONAS:
        if anchor_id in profiles:
            anchor_scores[anchor_id] = scorer.score_persona(profiles[anchor_id])
            print(f"  {anchor_id}: {anchor_scores[anchor_id]}")

    # Compute distances
    candidate_distances = {}
    for persona_id in candidates:
        if persona_id not in EXISTING_PERSONAS:
            dist = compute_distance_to_anchors(
                candidate_scores[persona_id],
                anchor_scores
            )
            candidate_distances[persona_id] = dist

    # Sort by distance (largest first = most different)
    sorted_candidates = sorted(
        candidate_distances.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"✓ Top 10 most distant candidates:")
    for i, (persona_id, dist) in enumerate(sorted_candidates[:10], 1):
        scores = candidate_scores[persona_id]
        print(f"  {i}. {persona_id}: dist={dist:.3f}, "
              f"F={scores['formality']:.2f}, E={scores['emotionality']:.2f}, "
              f"V={scores['verbosity']:.2f}, I={scores['interpersonal']:.2f}")

    # STEP 4: Assign archetypes and select 7 new personas
    print("\n" + "-" * 80)
    print("STEP 4: Assigning to target archetypes")
    print("-" * 80)

    archetype_assignments = {}
    for archetype in TARGET_ARCHETYPES:
        archetype_assignments[archetype] = []

    # Assign each candidate to best-matching archetype
    for persona_id, dist in sorted_candidates[:50]:  # Top 50 candidates
        if persona_id in EXISTING_PERSONAS:
            continue

        scores = candidate_scores[persona_id]
        examples = profiles[persona_id].get('example_responses', [])
        archetype = assign_archetype(scores, examples)

        archetype_assignments[archetype].append((persona_id, dist, scores))

    # Select best candidate for each archetype
    final_7_personas = []
    for archetype in TARGET_ARCHETYPES:
        candidates_for_arch = archetype_assignments[archetype]
        if candidates_for_arch:
            # Pick the one with largest distance
            best = max(candidates_for_arch, key=lambda x: x[1])
            final_7_personas.append({
                'persona_id': best[0],
                'archetype': archetype,
                'distance': best[1],
                'scores': best[2]
            })
            print(f"  ✓ {archetype}: {best[0]} (dist={best[1]:.3f})")
        else:
            print(f"  ✗ {archetype}: NO CANDIDATE FOUND")

    # STEP 5: Create final 10-persona list
    print("\n" + "-" * 80)
    print("STEP 5: Final 10-persona selection")
    print("-" * 80)

    final_10 = list(EXISTING_PERSONAS.keys()) + [p['persona_id'] for p in final_7_personas]

    print(f"\nFinal 10 personas:")
    for i, persona_id in enumerate(final_10, 1):
        if persona_id in EXISTING_PERSONAS:
            label = f"[EXISTING] {EXISTING_PERSONAS[persona_id]}"
            scores = anchor_scores[persona_id]
        else:
            match = next(p for p in final_7_personas if p['persona_id'] == persona_id)
            label = f"[NEW] {match['archetype']}"
            scores = match['scores']

        print(f"  P{i}. {persona_id}")
        print(f"      {label}")
        print(f"      F={scores['formality']:.2f}, E={scores['emotionality']:.2f}, "
              f"V={scores['verbosity']:.2f}, I={scores['interpersonal']:.2f}")

    # STEP 6: Save results
    print("\n" + "-" * 80)
    print("STEP 6: Saving results")
    print("-" * 80)

    output_dir = Path("/data01/nakata/master_thesis/persona2")

    # Save persona IDs
    with open(output_dir / "personas_final_10.txt", 'w') as f:
        for persona_id in final_10:
            f.write(f"{persona_id}\n")

    print(f"✓ Saved to: {output_dir / 'personas_final_10.txt'}")

    # Save detailed selection report
    report = {
        'selection_date': '2025-12-15',
        'total_candidates': len(profiles),
        'filtered_candidates': len(candidates),
        'existing_anchors': list(EXISTING_PERSONAS.keys()),
        'new_personas': [p['persona_id'] for p in final_7_personas],
        'final_10': final_10,
        'scores': {
            persona_id: candidate_scores.get(persona_id, anchor_scores.get(persona_id))
            for persona_id in final_10
        },
        'archetypes': {
            p['persona_id']: p['archetype'] for p in final_7_personas
        }
    }

    with open(output_dir / "persona_selection_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ Saved detailed report to: {output_dir / 'persona_selection_report.json'}")

    # Create LaTeX table for paper
    print("\n" + "-" * 80)
    print("LaTeX Table (for paper)")
    print("-" * 80)

    print("\n\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Selected Personas with Style Characteristics}")
    print("\\label{tab:persona_selection}")
    print("\\begin{tabular}{llcccc}")
    print("\\hline")
    print("ID & Archetype & F & E & V & I \\\\")
    print("\\hline")

    for i, persona_id in enumerate(final_10, 1):
        if persona_id in EXISTING_PERSONAS:
            archetype = EXISTING_PERSONAS[persona_id]
            scores = anchor_scores[persona_id]
        else:
            match = next(p for p in final_7_personas if p['persona_id'] == persona_id)
            archetype = match['archetype'].replace('_', ' ')
            scores = match['scores']

        short_id = persona_id.split('-')[1] + persona_id[-2:]
        print(f"P{i} & {archetype} & "
              f"{scores['formality']:.2f} & {scores['emotionality']:.2f} & "
              f"{scores['verbosity']:.2f} & {scores['interpersonal']:.2f} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    print("\n" + "=" * 80)
    print("✓ PERSONA SELECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
