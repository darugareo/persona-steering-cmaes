"""
Persona-Aware Judge for evaluating response alignment with individual personas.

This module implements the correct evaluation methodology:
- Judges responses based on INDIVIDUAL persona preferences
- Uses actual conversation history and behavioral patterns
- Measures persona-specific fit, NOT generic trait-likeness
"""

from typing import Dict, List, Any, Optional
import json
from pathlib import Path


def format_persona_profile_for_judge(profile: Dict[str, Any]) -> str:
    """
    Format persona profile into a concise summary for judge prompt.

    Args:
        profile: Persona profile dictionary from extract_persona_profiles.py

    Returns:
        Formatted string summarizing persona characteristics
    """
    persona_id = profile.get('persona_id', 'Unknown')
    num_sessions = profile.get('num_sessions', 0)

    # Communication style
    comm_style = profile.get('communication_style', {})
    style_desc = []
    if comm_style:
        avg_len = comm_style.get('avg_utterance_length', 0)
        if avg_len > 600:
            style_desc.append("tends to give detailed, lengthy responses")
        elif avg_len < 300:
            style_desc.append("prefers brief, concise responses")

        if comm_style.get('exclamation_rate', 0) > 0.5:
            style_desc.append("uses expressive punctuation frequently")

        if comm_style.get('first_person_plural_rate', 0) > comm_style.get('first_person_singular_rate', 0):
            style_desc.append("thinks in terms of 'we' rather than 'I'")
        elif comm_style.get('first_person_singular_rate', 0) > 2:
            style_desc.append("often uses first-person perspective")

        formality = comm_style.get('formality', 'neutral')
        style_desc.append(f"communication style is {formality}")

    # Relationship contexts
    relationships = profile.get('relationship_contexts', {})
    if relationships:
        main_context = max(relationships.items(), key=lambda x: x[1])[0]
        style_desc.append(f"primarily interacts in '{main_context}' context")

    # Values and priorities
    values = profile.get('values_and_priorities', {})
    priorities = values.get('inferred_priorities', [])

    # Compile summary
    summary = f"""**Persona ID:** {persona_id}
**Based on:** {num_sessions} conversation sessions

**Communication Style:**
{chr(10).join(f'- {s}' for s in style_desc) if style_desc else '- No specific patterns identified'}

**Values and Priorities:**
{chr(10).join(f'- {p}' for p in priorities) if priorities else '- Insufficient data to infer priorities'}
"""
    return summary


def format_example_responses(examples: List[Dict], max_examples: int = 5) -> str:
    """
    Format example responses from persona's conversation history.

    Args:
        examples: List of example response dictionaries
        max_examples: Maximum number of examples to include

    Returns:
        Formatted string with example responses
    """
    if not examples:
        return "No example responses available."

    formatted = []
    for i, example in enumerate(examples[:max_examples], 1):
        text = example.get('text', '')[:300]  # Truncate long examples
        context = example.get('relationship', 'unknown context')
        formatted.append(f"{i}. [{context}] {text}...")

    return "\n".join(formatted)


def create_persona_aware_judge_prompt(
    persona_profile: Dict[str, Any],
    original_prompt: str,
    response_a: str,
    response_b: str,
    comparison_type: str = "persona_fit"
) -> str:
    """
    Create persona-aware judge prompt.

    Args:
        persona_profile: Full persona profile dictionary
        original_prompt: Original user question/prompt
        response_a: First response to evaluate
        response_b: Second response to evaluate
        comparison_type: Type of comparison ("persona_fit" or "trait_specific")

    Returns:
        Complete judge prompt
    """
    # Format persona information
    persona_summary = format_persona_profile_for_judge(persona_profile)
    example_responses = format_example_responses(
        persona_profile.get('example_responses', []),
        max_examples=5
    )

    if comparison_type == "persona_fit":
        task_description = f"""You are evaluating which AI-generated response better matches a SPECIFIC INDIVIDUAL'S communication style and preferences.

This is NOT about generic "good" responses. This is about which response sounds more like something THIS SPECIFIC PERSON would say or prefer, based on their actual conversation history.
"""
        evaluation_criteria = """**Evaluation Criteria:**
1. **Communication Style Match:** Does the response match this person's typical way of expressing themselves?
2. **Value Alignment:** Does the response reflect this person's demonstrated priorities and values?
3. **Contextual Appropriateness:** Would this person give this type of response in this situation?
4. **Behavioral Consistency:** Is the response consistent with this person's past behavioral patterns?
"""
        output_format = """**Your Task:**
Compare Response A and Response B, and determine which one better aligns with THIS SPECIFIC PERSONA's characteristics.

Respond in JSON format:
{
  "winner": "A" or "B" or "tie",
  "confidence": <1-5, where 5 is very confident>,
  "explanation": "<2-3 sentences explaining why the chosen response better matches this persona's style>",
  "persona_fit_score_a": <1-5, how well Response A matches the persona>,
  "persona_fit_score_b": <1-5, how well Response B matches the persona>
}
"""
    else:  # trait_specific
        task_description = """You are evaluating which response better demonstrates a specific trait FOR THIS PARTICULAR PERSONA.

Consider both: (1) Does the response show the trait? (2) Is it consistent with how THIS PERSON expresses that trait?
"""
        evaluation_criteria = """**Evaluation Criteria:**
1. Does the response demonstrate the target trait?
2. Is the expression of the trait consistent with this persona's style?
3. Would this persona naturally express the trait in this way?
"""
        output_format = """**Your Task:**
Determine which response better demonstrates the trait while remaining consistent with the persona.

Respond in JSON format:
{
  "winner": "A" or "B" or "tie",
  "confidence": <1-5>,
  "explanation": "<brief explanation>",
  "trait_score_a": <1-5>,
  "trait_score_b": <1-5>
}
"""

    # Compile full prompt
    full_prompt = f"""{task_description}

{persona_summary}

**Example Responses from This Persona's Actual Conversations:**
{example_responses}

---

**User Question/Prompt:**
{original_prompt}

**Response A:**
{response_a}

**Response B:**
{response_b}

---

{evaluation_criteria}

{output_format}

Remember: You are NOT judging which response is "better" in general. You are judging which response is more aligned with THIS SPECIFIC PERSONA's demonstrated communication patterns and preferences.
"""

    return full_prompt


def create_persona_fit_scoring_prompt(
    persona_profile: Dict[str, Any],
    original_prompt: str,
    response: str
) -> str:
    """
    Create prompt for scoring single response's fit to persona.

    Args:
        persona_profile: Full persona profile dictionary
        original_prompt: Original user question
        response: Response to evaluate

    Returns:
        Judge prompt for scoring
    """
    persona_summary = format_persona_profile_for_judge(persona_profile)
    example_responses = format_example_responses(
        persona_profile.get('example_responses', []),
        max_examples=5
    )

    prompt = f"""You are evaluating how well an AI-generated response matches a SPECIFIC INDIVIDUAL'S communication style and preferences.

{persona_summary}

**Example Responses from This Persona's Actual Conversations:**
{example_responses}

---

**User Question:**
{original_prompt}

**AI-Generated Response:**
{response}

---

**Your Task:**
Rate how well this response matches THIS SPECIFIC PERSONA on a scale of 1-5:

1 = Completely inconsistent with this persona's style and preferences
2 = Somewhat misaligned with this persona
3 = Neutral / Could be this persona
4 = Mostly consistent with this persona's patterns
5 = Strongly matches this persona's communication style and values

Consider:
- Does it match their communication style (length, formality, expression)?
- Does it reflect their values and priorities?
- Is it consistent with how they've responded in similar contexts?

Respond in JSON format:
{{
  "persona_fit_score": <1-5>,
  "explanation": "<2-3 sentences explaining your rating>",
  "style_match": "<how well it matches communication style>",
  "value_match": "<how well it reflects their values>",
  "consistency": "<how consistent with past behavior>"
}}
"""

    return prompt


# Template registry for different evaluation scenarios
JUDGE_TEMPLATES = {
    "persona_comparison": create_persona_aware_judge_prompt,
    "persona_scoring": create_persona_fit_scoring_prompt,
}


def load_persona_profile(persona_id: str, profiles_dir: Path = Path("data/persona_profiles")) -> Dict[str, Any]:
    """
    Load persona profile from JSON file.

    Args:
        persona_id: Persona identifier
        profiles_dir: Directory containing persona profiles

    Returns:
        Persona profile dictionary
    """
    profile_path = profiles_dir / f"{persona_id}.json"

    if not profile_path.exists():
        raise FileNotFoundError(f"Persona profile not found: {profile_path}")

    with open(profile_path) as f:
        profile = json.load(f)

    return profile


def validate_judge_response(response: Dict[str, Any], expected_fields: List[str]) -> bool:
    """
    Validate that judge response contains expected fields.

    Args:
        response: Judge's JSON response
        expected_fields: List of required field names

    Returns:
        True if valid, False otherwise
    """
    return all(field in response for field in expected_fields)
