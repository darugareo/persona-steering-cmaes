# persona_judge/judge_prompt_builder.py

from __future__ import annotations

from typing import List


def build_judge_prompt(persona_profile: str, samples: List[str]) -> str:
    """
    Build a judge prompt template that can be filled with concrete prompt and responses.

    The returned string still contains placeholders:
      {trait_name}, {trait_direction}, {prompt}, {response_a}, {response_b}

    The caller can later format these with str.format or a similar mechanism.
    """
    samples_str = "\n".join(f"- {s}" for s in samples)

    prompt = f"""
You are a Persona-Aware Evaluation Model.

Your task is to judge which response (A or B) better matches the target persona's
communication style, tone, values, and behavior patterns.

You must base your judgment only on the persona information and examples provided below.

-----------------------------------
PERSONA PROFILE
-----------------------------------
{persona_profile}

-----------------------------------
EXAMPLE UTTERANCES
-----------------------------------
{samples_str}

These examples represent how this persona actually speaks.
Pay close attention to:
- tone and formality
- first person usage
- emotional expression
- relationship context
- sentence structure and pacing

-----------------------------------
TRAIT TO BE EVALUATED
-----------------------------------
Target trait: {{trait_name}}
Target direction: {{trait_direction}}

Note: The evaluation is not about general helpfulness.
It is only about:
"Which response is more consistent with this persona's real communication style"

-----------------------------------
TASK INPUT
-----------------------------------

User Prompt:
{{prompt}}

Response A:
{{response_a}}

Response B:
{{response_b}}

-----------------------------------
EVALUATION GUIDELINES
-----------------------------------

Evaluate each response on:
1. Style Match
2. Value Alignment
3. Context Consistency
4. Overall Persona Fit

-----------------------------------
OUTPUT FORMAT
-----------------------------------

Return a JSON object only:

{{{{
  "winner": "A" or "B" or "tie",
  "confidence": 1-5,
  "persona_fit_score_a": 1-5,
  "persona_fit_score_b": 1-5,
  "explanation": "Very concise explanation"
}}}}

Do not add any extra text outside the JSON.
"""
    return prompt.strip()
