# Prompt Design for Phase 1-B: Persona-Sensitive Evaluation

**Version**: v2
**Date**: 2025-12-15
**Total Prompts**: 28 (20 from Phase 1-A + 8 new for Phase 1-B)

---

## Design Philosophy

### Core Principle: "Persona-Agnostic but Persona-Sensitive"

**Persona-Agnostic**:
- No persona-specific names, relationships, or biographical details
- No factual questions with objective correct answers
- Universally applicable across all personas and models
- Content-neutral and context-independent

**Persona-Sensitive**:
- Elicit responses that vary significantly by persona style
- Target stylistic dimensions: tone, structure, emotional expression, interpersonal distance
- Allow human judges to distinguish persona-aligned vs. non-aligned responses
- Maximize discriminative power for A/B comparison

---

## Prompt Categories (7 Total)

### Phase 1-A Categories (IDs 1-20, Unchanged)

These 20 prompts were used in Phase 1-A generation and remain **completely unchanged** to maintain experimental continuity.

#### 1. Opinion (4 prompts)
**Purpose**: General viewpoint expression
**Stylistic Variation**: Assertiveness, hedging, confidence markers

- ID 1: Technology in modern education
- ID 6: Work-life balance
- ID 11: Importance of learning new skills
- ID 16: Value of breaks and rest

#### 2. Explanation (4 prompts)
**Purpose**: Conceptual clarity and pedagogical approach
**Stylistic Variation**: Simplicity vs. sophistication, use of examples, metaphors

- ID 2: Concept of fairness
- ID 7: Why reading is important
- ID 12: Concept of responsibility
- ID 17: How creativity works

#### 3. Advice (4 prompts)
**Purpose**: Guidance provision style
**Stylistic Variation**: Directiveness, empathy markers, prescriptive vs. suggestive

- ID 3: Starting a creative project
- ID 8: Feeling stuck in routine
- ID 13: Maintaining friendships
- ID 18: Dealing with difficult decisions

#### 4. Pros/Cons (4 prompts)
**Purpose**: Analytical balance and argumentation
**Stylistic Variation**: Structured lists vs. flowing prose, nuance vs. binary thinking

- ID 4: Working from home
- ID 9: Social media
- ID 14: Living in a big city
- ID 19: Competitive environments

#### 5. Reflection (4 prompts)
**Purpose**: Philosophical depth and introspection
**Stylistic Variation**: Abstract vs. concrete, self-reference, narrative richness

- ID 5: What success means
- ID 10: Define happiness
- ID 15: Role of failure in personal growth
- ID 20: Effort and results relationship

---

### Phase 1-B Additions (IDs 21-28, New)

These 8 prompts are specifically designed to **amplify persona-sensitive stylistic variation** for judge-based evaluation.

#### 6. Style-Sensitive Disagreement (4 prompts)
**Purpose**: Elicit interpersonal sensitivity and face-saving strategies
**Key Dimensions**:
- Politeness vs. directness
- Emotional tone (warm/neutral/cold)
- Hedging and mitigation strategies
- Interpersonal distance (formal/casual/intimate)

**Prompts**:

**ID 21**: *"Someone strongly disagrees with your opinion. How do you respond?"*
- **Target**: Conflict response style (confrontational vs. accommodating)
- **Expected Variation**: Direct rebuttal vs. empathetic acknowledgment vs. deflection

**ID 22**: *"You want to express disagreement without hurting the other person. What do you say?"*
- **Target**: Tact and face-saving language
- **Expected Variation**: Explicit softeners ("I understand, but...") vs. indirect suggestions vs. question-based challenges

**ID 23**: *"How do you explain a sensitive topic to someone close to you?"*
- **Target**: Intimacy calibration and disclosure depth
- **Expected Variation**: Emotional openness vs. protective distance vs. pedagogical framing

**ID 24**: *"When is it better to stay silent rather than speak up?"*
- **Target**: Wisdom framing and restraint philosophy
- **Expected Variation**: Pragmatic calculation vs. moral reflection vs. relational harmony focus

---

#### 7. Emotional Reflection (4 prompts)
**Purpose**: Elicit emotional expressiveness and vulnerability
**Key Dimensions**:
- Emotional vocabulary richness
- Metaphor and narrative detail
- Self-disclosure comfort level
- Growth mindset vs. protective framing

**Prompts**:

**ID 25**: *"Describe a moment when you felt disappointed but learned something important."*
- **Target**: Narrative style and emotional processing
- **Expected Variation**: Vivid storytelling vs. abstract lesson-extraction vs. minimization

**ID 26**: *"How do you usually react when plans suddenly change?"*
- **Target**: Adaptability framing and stress response
- **Expected Variation**: Frustration acknowledgment vs. optimistic reframing vs. control language ("I adjust quickly")

**ID 27**: *"What do you do when someone lets you down?"*
- **Target**: Disappointment processing and boundary-setting
- **Expected Variation**: Forgiveness emphasis vs. self-protection vs. relational repair strategies

**ID 28**: *"How do you talk about failure with someone you care about?"*
- **Target**: Vulnerability display and supportiveness
- **Expected Variation**: Normalizing ("everyone fails") vs. empathetic mirroring vs. solution-focused

---

## Why These Additions Matter for Judge Evaluation

### Problem with Phase 1-A Prompts Alone
While the original 20 prompts successfully elicit general stylistic differences (structure, formality, complexity), they may be **insufficient for fine-grained persona discrimination** in judge-based evaluation.

**Example Issue**:
- Prompt: "What's your opinion on technology in education?"
- Potential Problem: All personas might produce similarly balanced, informative responses
- Judge Difficulty: Hard to determine which response "better matches persona X"

### Solution: Phase 1-B Additions
The 8 new prompts target **high-variance stylistic dimensions** where personas are expected to diverge sharply:

1. **Disagreement Handling** (IDs 21-24):
   - Personas likely differ in politeness norms, conflict avoidance, directness
   - Judge can assess: "Does this response feel like how [Persona X] would handle disagreement?"

2. **Emotional Expression** (IDs 25-28):
   - Personas likely differ in emotional openness, narrative detail, vulnerability comfort
   - Judge can assess: "Does this emotional tone match [Persona X]'s communication style?"

---

## Experimental Design Rationale

### Constraint 1: No Persona-Specific Information
**Why**: All prompts must work universally across:
- All personas (episode-184019_A, episode-239427_A, episode-118328_B)
- All models (Mistral-7B, Llama-3-8B, future additions)
- All methods (base, prompt, equal, optimized)

**How Achieved**: Abstract, hypothetical framing without names/relationships

### Constraint 2: No Factual Correctness
**Why**: Judge evaluation should assess **style match**, not factual accuracy

**How Achieved**: Focus on opinions, advice, reflections—domains with no single correct answer

### Constraint 3: Maximum Discriminative Power
**Why**: Enable reliable A/B comparison ("Which response better matches Persona X?")

**How Achieved**:
- Target dimensions with known persona variation (from Chronicles training data)
- Avoid questions where all personas would respond identically

---

## Usage in Judge Experiments

### Phase 1-B Judge Protocol

**Input to Generation Model**:
- Prompt text only (no persona profile)
- Steering vectors applied (for equal/optimized methods)

**Input to Judge Model**:
- Persona profile (24 historical tweets showing style)
- Original prompt
- Response A (e.g., base method)
- Response B (e.g., optimized method)

**Judge Task**:
> "Which response (A or B) better matches the communication style shown in the persona profile?"

**Expected Outcome**:
- Phase 1-A prompts (1-20): General stylistic differences observable
- Phase 1-B prompts (21-28): **Sharp, human-perceptible** persona alignment differences
- Combined set (1-28): Robust statistical evaluation across diverse dimensions

---

## Validation Criteria

### Coverage
- [x] 7 distinct stylistic dimensions
- [x] 4 prompts per dimension (balanced)
- [x] 28 total prompts (sufficient for statistical power)

### Quality
- [x] Persona-agnostic (no specific names/relationships)
- [x] Persona-sensitive (high expected variation)
- [x] Judge-compatible (clear A/B comparison criteria)
- [x] Model-agnostic (works across architectures)

### Experimental Continuity
- [x] Phase 1-A prompts (1-20) unchanged
- [x] Phase 1-B additions (21-28) complement existing set
- [x] No conflicts or redundancy across categories

---

## Expected Persona Variation Examples

### Persona A (Structured, Analytical)
- **Disagreement**: Likely to use numbered points, logical structure
- **Emotion**: May downplay feelings, focus on lessons learned

### Persona B (Conversational, Questioning)
- **Disagreement**: Likely to use rhetorical questions, explore multiple perspectives
- **Emotion**: May use vivid language, metaphors, open emotional expression

### Persona C (Reflective, Balanced)
- **Disagreement**: Likely to emphasize understanding both sides, nuanced position
- **Emotion**: May frame experiences philosophically, balanced tone

*(These are hypothetical examples; actual variation emerges from Chronicles optimization)*

---

## Future Extensions

### Potential Phase 1-C Additions (If Needed)
If judge evaluation requires additional discriminative power:
- **Humor/Sarcasm**: "How do you handle situations that are frustrating but also a bit absurd?"
- **Persuasion**: "How would you convince someone to reconsider a firmly held belief?"
- **Cultural Sensitivity**: "How do you approach discussions with people from very different backgrounds?"

### Multilingual Extension
Current prompts are English-only. For multilingual persona evaluation:
- Translate prompts maintaining abstract/hypothetical framing
- Ensure cultural neutrality in topic selection

---

## Conclusion

The **28-prompt set (v2)** provides:

1. **Experimental Continuity**: Phase 1-A results remain valid
2. **Enhanced Discrimination**: Phase 1-B additions target high-variance dimensions
3. **Judge Readiness**: All prompts suitable for A/B persona-match evaluation
4. **Research Contribution**: Demonstrates persona steering across diverse communicative contexts

**Status**: Ready for Phase 1-B judge evaluation experiments.

---

## File Locations

- **JSON**: `data/eval_prompts/persona_eval_prompts_v2.json`
- **Phase 1-A Originals**: `experiments/prompts/cross_model_prompts_v1.json` (unchanged)
- **This Document**: `docs/PROMPT_DESIGN_PHASE1B.md`

---

**Prepared by**: Claude Code
**Experiment Phase**: 1-B Preparation
**Next Step**: Judge evaluation with 28-prompt set
