# persona_judge/utils.py

from __future__ import annotations

from collections import Counter
from typing import List, Dict


def count_first_person(utterances: List[str]) -> tuple[int, int]:
    """
    Count rough first person singular and plural usage per utterance.
    This is very language specific and heuristic.

    For Japanese:
      singular: "私", "俺", "僕"
      plural: "私たち", "俺ら", "僕ら", "うちら"
    For English:
      singular: " I "
      plural: " we "
    """
    singular_tokens = ["私", "俺", "僕", " I "]
    plural_tokens = ["私たち", "俺ら", "僕ら", "うちら", " we "]

    singular = 0
    plural = 0
    for u in utterances:
        for t in singular_tokens:
            singular += u.count(t)
        for t in plural_tokens:
            plural += u.count(t)

    return singular, plural


def detect_formality_level(utterances: List[str]) -> str:
    """
    Very simple formality classifier: "informal", "formal", "neutral".
    Heuristics for Japanese.
    """
    if not utterances:
        return "unknown"

    text = "\n".join(utterances)

    informal_markers = ["だよ", "だな", "じゃん", "w", "笑", "ねえ", "かなあ", "だろ"]
    formal_markers = ["です", "ます", "でしょう", "ございます"]

    informal_score = sum(text.count(m) for m in informal_markers)
    formal_score = sum(text.count(m) for m in formal_markers)

    if informal_score > formal_score * 1.5 and informal_score > 3:
        return "informal"
    if formal_score > informal_score * 1.5 and formal_score > 3:
        return "formal"
    return "neutral"


def detect_relationship_context(utterances: List[str]) -> Dict[str, int]:
    """
    Rough count of relationship contexts from keywords.
    """
    text = "\n".join(utterances)

    patterns = {
        "Husband and Wife": ["夫", "妻", "嫁", "旦那", "結婚", "奥さん", "旦那さん"],
        "Friends": ["友達", "友だち", "親友"],
        "Family": ["家族", "兄弟", "姉", "弟", "妹", "両親"],
        "Work": ["上司", "同僚", "部下", "仕事", "職場"],
        "Neighbors": ["近所", "隣", "隣人"],
    }

    counts: Dict[str, int] = {}
    for label, keys in patterns.items():
        c = sum(text.count(k) for k in keys)
        if c > 0:
            counts[label] = c

    if not counts:
        counts["Unknown"] = 1

    return counts
