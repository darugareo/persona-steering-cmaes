# persona_judge/feature_extractor.py

from __future__ import annotations

from collections import Counter
from typing import List, Dict, Any

from .utils import count_first_person, detect_formality_level, detect_relationship_context


def extract_persona_features(conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract simple style features from conversation logs.

    This is intentionally heuristic and lightweight but deterministic.
    """
    if not conversations:
        raise ValueError("No conversations provided to extract_persona_features")

    # collect all user and assistant utterances together for style statistics
    utterances = [c["content"] for c in conversations if isinstance(c.get("content"), str)]
    lengths = [len(u) for u in utterances]

    avg_len = sum(lengths) / len(lengths) if lengths else 0.0

    exclam_count = sum(u.count("!") for u in utterances)
    question_count = sum(u.count("?") for u in utterances)

    first_sg_count, first_pl_count = count_first_person(utterances)

    style = detect_formality_level(utterances)
    rel_context_counts = detect_relationship_context(utterances)

    total_utts = len(utterances) or 1
    features: Dict[str, Any] = {
        "num_utterances": len(utterances),
        "avg_utterance_length": avg_len,
        "exclamation_rate": exclam_count / total_utts,
        "question_rate": question_count / total_utts,
        "first_person_singular_rate": first_sg_count / total_utts,
        "first_person_plural_rate": first_pl_count / total_utts,
        "formality": style,
        "relationship_contexts": rel_context_counts,
    }

    # rough additional behavioral hints
    behavior = _infer_behavioral_tendencies(utterances)
    features["behavioral_tendencies"] = behavior

    return features


def _infer_behavioral_tendencies(utterances: List[str]) -> Dict[str, float]:
    """
    Very rough heuristics for behavioral tendencies.
    Returns scores between 0 and 1.
    """
    if not utterances:
        return {
            "empathy": 0.0,
            "assertiveness": 0.0,
            "humor": 0.0,
            "directness": 0.0,
        }

    text = "\n".join(utterances)

    empathy_keywords = ["大丈夫", "つらい", "しんどい", "心配", "寄り添", "辛い", "頑張", "support", "sorry", "feel"]
    assertive_keywords = ["べき", "だろ", "決める", "やる", "するべき"]
    humor_keywords = ["笑", "w", "草", "lol", "笑った"]
    direct_keywords = ["正直", "ぶっちゃけ", "率直", "はっきり", "直接"]

    def score_for(words: List[str]) -> float:
        if not words:
            return 0.0
        c = sum(text.count(w) for w in words)
        # simple saturating normalization
        return min(1.0, c / 20.0)

    return {
        "empathy": score_for(empathy_keywords),
        "assertiveness": score_for(assertive_keywords),
        "humor": score_for(humor_keywords),
        "directness": score_for(direct_keywords),
    }
