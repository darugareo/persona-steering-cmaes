# persona_judge/persona_profile.py

from __future__ import annotations

from typing import Dict, Any, List


def generate_persona_profile(features: Dict[str, Any], samples: List[str]) -> str:
    """
    Generate a natural language persona profile from extracted features and sample utterances.
    The profile is concise but descriptive, intended for a judge prompt.
    """
    avg_len = features.get("avg_utterance_length", 0.0)
    formality = features.get("formality", "unknown")
    ex_rate = features.get("exclamation_rate", 0.0)
    q_rate = features.get("question_rate", 0.0)
    fp_sg = features.get("first_person_singular_rate", 0.0)
    fp_pl = features.get("first_person_plural_rate", 0.0)
    rel_ctx = features.get("relationship_contexts", {})
    behavior = features.get("behavioral_tendencies", {})

    rel_desc = ", ".join(f"{k}: {v}" for k, v in rel_ctx.items()) or "no dominant context"
    empathy = behavior.get("empathy", 0.0)
    assertive = behavior.get("assertiveness", 0.0)
    humor = behavior.get("humor", 0.0)
    direct = behavior.get("directness", 0.0)

    lines: List[str] = []

    lines.append("This persona is defined from past conversations with the model.")
    lines.append(f"Overall formality: {formality}")
    lines.append(f"Average utterance length: {avg_len:.1f} characters per message.")
    lines.append(
        f"First person usage: singular {fp_sg:.2f} per message, plural {fp_pl:.2f} per message."
    )
    lines.append(
        f"Punctuation tendencies: exclamation rate {ex_rate:.2f}, question rate {q_rate:.2f} per message."
    )
    lines.append(f"Relationship contexts observed: {rel_desc}.")
    lines.append(
        "Behavioral tendencies (0 to 1 scale): "
        f"empathy {empathy:.2f}, assertiveness {assertive:.2f}, humor {humor:.2f}, directness {direct:.2f}."
    )

    if formality == "informal":
        lines.append("The persona usually speaks in an informal, relaxed tone.")
    elif formality == "formal":
        lines.append("The persona usually speaks in a more polite and formal tone.")
    else:
        lines.append("The persona mixes informal and neutral expressions.")

    if humor > 0.2:
        lines.append("Humor or light joking appears from time to time.")
    if empathy > 0.2:
        lines.append("The persona often reacts with empathy and emotional awareness.")
    if direct > 0.2:
        lines.append("The persona tends to be relatively direct or candid in phrasing.")

    # add a short note on how to use samples
    if samples:
        lines.append(
            "The following example utterances should be treated as ground truth for this persona's style, "
            "including tone, structure, and typical content."
        )

    return "\n".join(lines)
