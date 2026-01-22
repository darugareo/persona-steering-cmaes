# Evaluation Metrics for Persona-Aware Steering

## Overview

This document defines the evaluation metrics used in the persona-aware steering project. It clearly distinguishes between two fundamentally different types of evaluation:

1. **Technical Validation**: Does the steering mechanism work?
2. **Persona Validation**: Does steering improve individual persona reproduction?

## Critical Distinction

```
⚠️ HIGH WIN RATE ≠ GOOD PERSONA FIT

Example:
- Generic Trait Score: 4.5/5 → "Response shows other-focus"
- Persona Fit Score: 2.0/5 → "Response doesn't match this person's informal style"

The response may demonstrate the trait well, but fail to match the individual's
communication patterns, values, and behavioral preferences.
```

## Metric Definitions

### 1. Generic Trait Metrics (Technical Validation Only)

**Purpose**: Validate that steering affects model outputs in the intended direction.

**Metrics**:
- `generic_trait_score` (1-5): How well the response demonstrates the target trait in general
- `trait_direction_accuracy`: Whether steering moves output toward the intended trait direction
- `steering_effect_magnitude`: Difference between baseline and steered responses

**Evaluation Methodology**:
```python
# Judge receives ONLY:
# - Response text
# - Trait name and direction
# NO persona context

judge_prompt = f"""
Rate how well this response demonstrates {trait_name} ({trait_direction})
on a scale of 1-5.

Response: {response}
"""
```

**Use Cases**:
- ✅ Verifying steering implementation works
- ✅ Comparing different steering methods
- ✅ Layer sweep technical validation

**Limitations**:
- ❌ Does NOT validate persona-specific fit
- ❌ Does NOT measure individual preference alignment
- ❌ Does NOT consider communication style matching

---

### 2. Persona-Specific Metrics (Persona Validation)

**Purpose**: Measure how well steered responses match specific individuals' preferences and patterns.

**Metrics**:

#### Primary Metric
- `persona_fit_score` (1-5): Overall alignment with THIS specific persona's style and preferences
  - 1 = Completely inconsistent with persona
  - 2 = Somewhat misaligned
  - 3 = Neutral / Could be this persona
  - 4 = Mostly consistent
  - 5 = Strongly matches persona

#### Detailed Sub-Metrics
- `style_match` (text): How well response matches communication style
  - Considers: utterance length, formality, punctuation, pronoun usage
- `value_match` (text): How well response reflects persona's demonstrated values
  - Considers: inferred priorities, behavioral patterns
- `consistency` (text): Consistency with past behavior in similar contexts
  - Considers: relationship context, example responses

#### Comparative Metrics
- `persona_improvement`: `steered_fit - baseline_fit`
- `persona_win_rate`: Percentage where steered response better matches persona
- `persona_alignment_rate`: Percentage where steered achieves fit score ≥ 4

**Evaluation Methodology**:
```python
# Judge receives FULL PERSONA CONTEXT:
# - Persona profile (communication style, values, priorities)
# - Example responses from actual conversations
# - Relationship contexts
# - Behavioral patterns

judge_prompt = create_persona_fit_scoring_prompt(
    persona_profile=persona_profile,  # Full profile
    original_prompt=original_prompt,
    response=response
)
```

**Use Cases**:
- ✅ Validating persona reproduction improvement
- ✅ Selecting optimal steering vectors per individual
- ✅ Measuring personalization quality

**Requirements**:
- ✅ Must have extracted persona profiles
- ✅ Must include example responses from actual conversations
- ✅ Must provide judge with full behavioral context

---

## Evaluation Pipeline Stages

### Stage 1: Technical Validation ✅ COMPLETED

**Goal**: Verify steering mechanism affects outputs directionally.

**Metrics Used**: Generic trait metrics

**Validation Criteria**:
- [ ] Steering changes outputs (not identical to baseline)
- [ ] Direction is as intended (toward target trait)
- [ ] Effect scales with alpha parameter
- [ ] Consistent across multiple prompts

**Status**: ✅ Validated in `reports/layer_sweep_summary.md`

**Key Finding**: Steering works technically - responses do shift toward "other-focus" trait.

---

### Stage 2: Persona Validation ⚠️ IN PROGRESS

**Goal**: Verify steering improves individual persona reproduction.

**Metrics Used**: Persona-specific metrics

**Validation Criteria**:
- [ ] Steered responses better match individual communication styles
- [ ] Improvement generalizes across different personas
- [ ] Optimal steering parameters vary by persona
- [ ] Persona fit scores ≥ 4 for most responses

**Status**: ⚠️ Script implemented, initial testing done, full validation pending

**Current Finding**: Initial test shows both baseline and steered have low persona fit (2/5) for informal conversational persona, indicating need for persona-specific steering vectors.

---

## Reporting Requirements

### Technical Reports (Stage 1)

Must include:
- ✅ Generic trait scores and win rates
- ✅ Layer sweep results
- ✅ Alpha sensitivity analysis
- ⚠️ **Prominent disclaimer**: "Technical validation only - persona fit not validated"

### Persona Validation Reports (Stage 2)

Must include:
- Persona-specific fit scores per individual
- Breakdown by communication style dimensions
- Comparison to baseline persona fit
- Identification of cases where steering helps vs. hurts persona fit

Must **separately report**:
- Generic trait metrics (technical)
- Persona-specific metrics (validation goal)

**Forbidden**: Conflating technical win rate with persona fit improvement.

---

## Example Evaluation Comparison

### Scenario: Evaluating "other-focused" steering on episode-184019_A

#### Generic Trait Evaluation (Stage 1)
```json
{
  "generic_trait_score": 4.2,
  "explanation": "Response demonstrates empathy and considers others' needs",
  "technical_validation": "PASS"
}
```

#### Persona-Specific Evaluation (Stage 2)
```json
{
  "persona_fit_score": 2.0,
  "style_match": "Formal and structured, contrasts with persona's informal and casual style",
  "value_match": "Does not reflect specific values or priorities of the persona",
  "consistency": "Does not align with past behavior (informal, personal anecdotes)",
  "persona_validation": "FAIL"
}
```

**Interpretation**:
- ✅ Steering works technically (high generic trait score)
- ❌ Steering does NOT help this individual's persona (low fit score)
- 🔧 Need persona-specific steering vectors

---

## Implementation Files

### Generic Trait Evaluation
- `scripts/eval_layer_sweep_l3.py` (original)
- Judge template: Simple trait-based prompts

### Persona-Specific Evaluation
- `scripts/eval_layer_sweep_persona_aware.py`
- `persona_opt/persona_aware_judge.py`
- Judge templates: `create_persona_fit_scoring_prompt()`, `create_persona_aware_judge_prompt()`
- Persona profiles: `data/persona_profiles/*.json`

---

## Future Metrics (Planned)

### Multi-Trait Persona Metrics
Once multi-trait steering is implemented (Task 5):

- `trait_weight_optimization`: Optimal w1-w5 weights per persona
- `multi_trait_fit_score`: Persona fit using weighted combination of trait vectors
- `persona_specificity`: How much optimal weights differ across personas

### Large-Scale Validation Metrics
Once evaluation set is expanded (Task 6):

- `cross_prompt_consistency`: Persona fit consistency across 20-50 prompts
- `persona_generalization`: Fit on held-out prompts
- `inter_persona_variance`: How much optimal steering differs across individuals

---

## Summary

| Metric Type | Purpose | Judge Receives | Use Case | Status |
|-------------|---------|----------------|----------|--------|
| Generic Trait | Technical validation | Response + trait name | Verify steering works | ✅ Done |
| Persona-Specific | Persona reproduction | Response + full persona profile | Validate personalization | ⚠️ In progress |

**Key Principle**: Never conflate technical validation with persona validation. They measure fundamentally different things.
