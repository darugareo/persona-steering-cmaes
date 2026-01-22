# persona_judge/sample_selector.py

from __future__ import annotations

from typing import List, Dict, Any


def select_representative_samples(
    conversations: List[Dict[str, Any]],
    n: int = 10,
    max_len: int = 300,
) -> List[str]:
    """
    Select representative utterances for the persona.

    Strategy:
      1. Prefer user role utterances, then assistant.
      2. Filter out too short or too long content.
      3. Take up to n diverse samples in original order.
    """
    candidates: List[str] = []

    for role in ("user", "assistant"):
        for item in conversations:
            if item.get("role") != role:
                continue
            content = item.get("content")
            if not isinstance(content, str):
                continue
            length = len(content)
            if length < 20:
                continue
            if length > max_len:
                continue
            candidates.append(content)
            if len(candidates) >= n:
                break
        if len(candidates) >= n:
            break

    # fallback: if still not enough, just add remaining truncated utterances
    if len(candidates) < n:
        for item in conversations:
            content = item.get("content")
            if not isinstance(content, str):
                continue
            if content in candidates:
                continue
            truncated = content[:max_len]
            candidates.append(truncated)
            if len(candidates) >= n:
                break

    return candidates[:n]
