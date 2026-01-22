#!/usr/bin/env python3
"""
Complete Setup for 26 Personas
================================

This script creates all necessary files for 26 personas:
1. Extract raw conversations from dataset
2. Generate persona_features.json
3. Generate persona_samples.json
4. Generate persona_profile.txt (for judge)
5. Generate final_judge_prompt.txt (for judge)

Input:
  - personas_final_26.txt
  - data/processed/cc/*.parquet

Output:
  - personas/{persona_id}/*.json, *.txt
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment")

client = OpenAI(api_key=OPENAI_API_KEY)

# Paths
PERSONAS_FILE = Path("personas_final_26.txt")
BASE_DIR = Path("personas")
DATA_DIR = Path("data/processed/cc")

# Load persona list
print("=" * 80)
print("COMPLETE SETUP FOR 26 PERSONAS")
print("=" * 80)

with open(PERSONAS_FILE) as f:
    PERSONAS_26 = [line.strip() for line in f if line.strip()]

print(f"\n✓ Loaded {len(PERSONAS_26)} personas from {PERSONAS_FILE}")

# Step 1: Extract raw conversations
print("\n" + "=" * 80)
print("STEP 1: Extract Raw Conversations")
print("=" * 80)

def extract_raw_conversations(persona_id: str) -> Dict:
    """Extract raw conversations from parquet files"""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    # Load session docs
    session_docs = pq.read_table(DATA_DIR / "persona_session_docs.parquet")

    # Filter by persona_id using pyarrow compute
    mask = pc.equal(session_docs['persona_id'], persona_id)
    persona_sessions = session_docs.filter(mask)

    if persona_sessions.num_rows == 0:
        print(f"  ⚠️  No sessions found for {persona_id}")
        return {"persona_id": persona_id, "sessions": []}

    # Convert to dict
    sessions = []
    for i in range(persona_sessions.num_rows):
        row = {col: persona_sessions[col][i].as_py() for col in persona_sessions.column_names}
        sessions.append(row)

    return {
        "persona_id": persona_id,
        "num_sessions": len(sessions),
        "sessions": sessions
    }

# Step 2: Generate persona_features.json
def generate_persona_features(raw_conversations: Dict) -> Dict:
    """Generate statistical features from raw conversations"""
    sessions = raw_conversations.get("sessions", [])

    if not sessions:
        return {}

    # Collect all utterances
    all_text = []
    for session in sessions:
        text = session.get("session_text", "")
        if text:
            all_text.append(text)

    combined_text = " ".join(all_text)

    # Count stats
    utterances = [u for u in combined_text.split('\n') if u.strip()]
    num_utterances = len(utterances)

    if num_utterances == 0:
        return {}

    avg_length = sum(len(u) for u in utterances) / num_utterances

    # Count punctuation
    exclamation_count = sum(u.count('!') for u in utterances)
    question_count = sum(u.count('?') for u in utterances)

    # Count first person
    text_lower = combined_text.lower()
    first_person_sg = len(re.findall(r'\b(i|me|my|myself)\b', text_lower))
    first_person_pl = len(re.findall(r'\b(we|us|our|ourselves)\b', text_lower))

    # Estimate formality (simple heuristic)
    formal_words = ['please', 'thank you', 'appreciate', 'kindly', 'regards']
    informal_words = ['yeah', 'yep', 'nope', 'gonna', 'wanna', 'kinda']

    formal_count = sum(text_lower.count(w) for w in formal_words)
    informal_count = sum(text_lower.count(w) for w in informal_words)

    if informal_count > formal_count:
        formality = "informal"
    elif formal_count > informal_count:
        formality = "formal"
    else:
        formality = "neutral"

    # Behavioral tendencies (simple heuristics)
    empathy_words = ['understand', 'feel', 'sorry', 'care', 'support']
    humor_words = ['haha', 'lol', 'funny', 'joke', 'laugh']

    empathy_score = min(1.0, sum(text_lower.count(w) for w in empathy_words) / max(num_utterances, 1) * 2)
    humor_score = min(1.0, sum(text_lower.count(w) for w in humor_words) / max(num_utterances, 1) * 5)

    return {
        "num_utterances": num_utterances,
        "avg_utterance_length": avg_length,
        "exclamation_rate": exclamation_count / num_utterances,
        "question_rate": question_count / num_utterances,
        "first_person_singular_rate": first_person_sg / num_utterances,
        "first_person_plural_rate": first_person_pl / num_utterances,
        "formality": formality,
        "relationship_contexts": {"Unknown": 1},
        "behavioral_tendencies": {
            "empathy": round(empathy_score, 2),
            "assertiveness": 0.0,
            "humor": round(humor_score, 2),
            "directness": 0.0
        }
    }

# Step 3: Generate persona_samples.json
def generate_persona_samples(raw_conversations: Dict, n_samples: int = 5) -> Dict:
    """Select representative samples from conversations"""
    sessions = raw_conversations.get("sessions", [])

    if not sessions:
        return {"samples": []}

    samples = []

    # Sample from different sessions
    import random
    random.seed(42)

    sampled_sessions = random.sample(sessions, min(n_samples, len(sessions)))

    for session in sampled_sessions:
        text = session.get("session_text", "").strip()
        relationship = session.get("relationship", "Unknown")

        # Take first 300 chars as sample
        if len(text) > 300:
            text = text[:300] + "..."

        if text:
            samples.append({
                "text": text,
                "relationship": relationship
            })

    return {"samples": samples}

# Step 4: Generate persona_profile.txt
def generate_persona_profile_txt(features: Dict) -> str:
    """Generate text profile for judge (same format as existing personas)"""
    if not features:
        return "Profile generation failed: no features available"

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

    lines = []
    lines.append("This persona is defined from past conversations with the model.")
    lines.append(f"Overall formality: {formality}")
    lines.append(f"Average utterance length: {avg_len:.1f} characters per message.")
    lines.append(f"First person usage: singular {fp_sg:.2f} per message, plural {fp_pl:.2f} per message.")
    lines.append(f"Punctuation tendencies: exclamation rate {ex_rate:.2f}, question rate {q_rate:.2f} per message.")
    lines.append(f"Relationship contexts observed: {rel_desc}.")
    lines.append(f"Behavioral tendencies (0 to 1 scale): empathy {empathy:.2f}, assertiveness {assertive:.2f}, humor {humor:.2f}, directness {direct:.2f}.")

    if formality == "informal":
        lines.append("The persona usually speaks in an informal, relaxed tone.")
    elif formality == "formal":
        lines.append("The persona usually speaks in a more polite and formal tone.")

    if humor > 0.2:
        lines.append("Humor or light joking appears from time to time.")
    if empathy > 0.2:
        lines.append("The persona often reacts with empathy and emotional awareness.")

    lines.append("The following example utterances should be treated as ground truth for this persona's style, including tone, structure, and typical content.")

    return "\n".join(lines)

# Step 5: Generate final_judge_prompt.txt
JUDGE_TEMPLATE = """You are a Persona-Aware Evaluation Model.

Your task is to judge which response (A or B) better matches the target persona's
communication style, tone, values, and behavior patterns.

You must base your judgment only on the persona information and examples provided below.

-----------------------------------
PERSONA PROFILE
-----------------------------------
{profile}

-----------------------------------
EXAMPLE UTTERANCES
-----------------------------------
{examples}

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

Return your evaluation in the following JSON format:
{{{{
  "winner": "A" or "B" or "tie",
  "confidence": 1-5,
  "persona_fit_score_a": 1-5,
  "persona_fit_score_b": 1-5,
  "explanation": "Brief explanation of your decision"
}}}}
"""

def generate_judge_prompt(profile_txt: str, samples) -> str:
    """Generate judge prompt from profile and samples"""
    # Handle both dict format (new) and list format (old)
    if isinstance(samples, dict):
        samples_list = samples.get("samples", [])
    else:
        # Old format: list of strings
        samples_list = [{"text": s} for s in (samples if isinstance(samples, list) else [])]

    # Format examples
    examples_list = []
    for sample in samples_list:
        if isinstance(sample, str):
            text = sample
        else:
            text = sample.get("text", "")

        # Extract individual utterances (split by newlines)
        utterances = [u.strip() for u in text.split('\n') if u.strip()]
        examples_list.extend(utterances[:3])  # Max 3 per sample

    # Limit to 10 examples total
    examples_list = examples_list[:10]

    examples_text = "\n".join(f"- {ex}" for ex in examples_list)

    return JUDGE_TEMPLATE.format(
        profile=profile_txt,
        examples=examples_text
    )

# Main processing loop
print("\nProcessing personas...")

for i, persona_id in enumerate(PERSONAS_26, 1):
    print(f"\n{'=' * 80}")
    print(f"[{i}/{len(PERSONAS_26)}] {persona_id}")
    print(f"{'=' * 80}")

    persona_dir = BASE_DIR / persona_id
    persona_dir.mkdir(exist_ok=True, parents=True)

    # Check existing files
    existing_files = list(persona_dir.glob("*"))
    print(f"  Existing files: {len(existing_files)}")

    # Step 1: Raw conversations
    raw_conv_path = persona_dir / "raw_conversations.json"
    if raw_conv_path.exists():
        print(f"  ✓ raw_conversations.json exists (skipping extraction)")
        with open(raw_conv_path) as f:
            raw_conversations = json.load(f)
    else:
        print(f"  Extracting raw conversations...")
        try:
            raw_conversations = extract_raw_conversations(persona_id)
            with open(raw_conv_path, 'w') as f:
                json.dump(raw_conversations, f, indent=2, ensure_ascii=False)
            print(f"    ✓ Saved: raw_conversations.json ({raw_conversations.get('num_sessions', 0)} sessions)")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            raw_conversations = {"persona_id": persona_id, "sessions": []}

    # Step 2: Features
    features_path = persona_dir / "persona_features.json"
    if features_path.exists():
        print(f"  ✓ persona_features.json exists (skipping)")
        with open(features_path) as f:
            features = json.load(f)
    else:
        print(f"  Generating persona_features.json...")
        features = generate_persona_features(raw_conversations)
        with open(features_path, 'w') as f:
            json.dump(features, f, indent=2, ensure_ascii=False)
        print(f"    ✓ Saved: persona_features.json")

    # Step 3: Samples
    samples_path = persona_dir / "persona_samples.json"
    if samples_path.exists():
        print(f"  ✓ persona_samples.json exists (skipping)")
        with open(samples_path) as f:
            samples = json.load(f)
    else:
        print(f"  Generating persona_samples.json...")
        samples = generate_persona_samples(raw_conversations)
        with open(samples_path, 'w') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"    ✓ Saved: persona_samples.json ({len(samples.get('samples', []))} samples)")

    # Step 4: Profile TXT
    profile_txt_path = persona_dir / "persona_profile.txt"
    if profile_txt_path.exists():
        print(f"  ✓ persona_profile.txt exists (skipping)")
        profile_txt = profile_txt_path.read_text()
    else:
        print(f"  Generating persona_profile.txt...")
        profile_txt = generate_persona_profile_txt(features)
        with open(profile_txt_path, 'w') as f:
            f.write(profile_txt)
        print(f"    ✓ Saved: persona_profile.txt ({len(profile_txt)} chars)")

    # Step 5: Judge prompt
    judge_prompt_path = persona_dir / "final_judge_prompt.txt"
    if judge_prompt_path.exists():
        print(f"  ✓ final_judge_prompt.txt exists (skipping)")
    else:
        print(f"  Generating final_judge_prompt.txt...")
        judge_prompt = generate_judge_prompt(profile_txt, samples)
        with open(judge_prompt_path, 'w') as f:
            f.write(judge_prompt)
        print(f"    ✓ Saved: final_judge_prompt.txt ({len(judge_prompt)} chars)")

# Verification
print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)

required_files = [
    "raw_conversations.json",
    "persona_features.json",
    "persona_samples.json",
    "persona_profile.txt",
    "final_judge_prompt.txt"
]

all_complete = True

for persona_id in PERSONAS_26:
    persona_dir = BASE_DIR / persona_id
    missing = []

    for filename in required_files:
        if not (persona_dir / filename).exists():
            missing.append(filename)

    if missing:
        print(f"  ✗ {persona_id}: Missing {missing}")
        all_complete = False
    else:
        print(f"  ✓ {persona_id}: Complete")

if all_complete:
    print("\n" + "=" * 80)
    print("✓ ALL 26 PERSONAS COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review generated profiles")
    print("  2. Run optimization experiments:")
    print("     python scripts/run_persona_optimization.py --persona_id <persona_id>")
else:
    print("\n⚠️  Some personas are incomplete. Check errors above.")
