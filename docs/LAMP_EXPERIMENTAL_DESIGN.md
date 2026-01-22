# LaMP Experimental Design & Assumptions

## Overview

This document outlines the experimental design for evaluating training-free persona steering using the LaMP benchmark.

## LaMP-7 Task Definition

### Official Task Description

**Source**: Salemi et al. (2023), "LaMP: When Large Language Models Meet Personalization", arXiv:2304.11406

**Task Name**: LaMP-7 — Personalized Tweet Paraphrasing

**Task Description**:
> "Social media posts adhere strongly to various personal stylistic patterns of authors [...] generate a tweet in the style of a user given an input tweet x, and a user profile of historical tweets by the user."

**Dataset Source**: Sentiment140 dataset (Go et al., 2009)

### Formal Task Specification

**Input (x)**: A tweet to be paraphrased
- Format: `"Paraphrase the following tweet without any explanation before or after it: {original_tweet}"`
- Example: `"Paraphrase the following tweet without any explanation before or after it: I'm currently enjoying the album \"Listen to Eason Chan.\""`

**Output (y)**: A paraphrased version of the input tweet that preserves semantic meaning while reflecting the user's personal writing style
- Example input: `"I'm currently enjoying the album \"Listen to Eason Chan.\""`
- Example output: `"Listening to \"Listen to Eason Chan\" it's a good album "`

**User Profile (Pᵤ)**: A collection of 24 historical tweets authored by the target user
- Each profile entry contains: `{"text": "<tweet_content>", "id": "<unique_id>"}`
- Profile provides stylistic patterns: tone, vocabulary, punctuation, emoji usage, grammatical patterns

**Official Evaluation Metrics**: ROUGE-1 and ROUGE-L (as specified in Section 2.3 of the paper)

**Data Split**: User-based split (testing personalization for new users not seen during training)

### Task Characteristics

1. **Single-turn generation**: Each sample is an independent paraphrasing task
2. **Style-focused**: Requires matching user's linguistic patterns, not just semantic equivalence
3. **Profile-driven**: User's writing style is encoded in 24 historical tweets (~2,200 characters total)
4. **Constrained generation**: Must paraphrase (preserve meaning) while adapting style

---

## Our Experimental Protocol for LaMP-7

### Critical Design Decision: Profile Isolation

**Generation Phase (Model Input)**:
- ✅ Input x (task prompt with tweet to paraphrase)
- ✅ Steering vectors (trait weights from Chronicles optimization)
- ❌ User profile Pᵤ (NOT provided to model)

**Evaluation Phase (Judge Input)**:
- ✅ Input x (original task prompt)
- ✅ Generated output ŷ
- ✅ User profile Pᵤ (24 historical tweets)
- ✅ Reference output y (gold paraphrase)

### Rationale

**Why profile is excluded from generation**:
1. Tests whether trait vectors learned on Conversation Chronicles generalize to LaMP users
2. Prevents "profile memorization" — model must rely solely on steering vectors
3. Evaluates true transfer of persona characteristics across domains
4. Maintains scientific validity: optimization dataset (Chronicles) ≠ evaluation dataset (LaMP)

**Why profile is shown to judge**:
1. Judge needs ground truth to assess style consistency
2. Mirrors real-world scenario: evaluator has access to user history
3. Enables fair comparison: all methods evaluated against same user style baseline

### Research Question

**Primary hypothesis**: Trait vectors optimized on Chronicles conversations will produce outputs that better match LaMP user styles (as judged by profile comparison) compared to baselines.

**This tests**: Domain transfer of persona steering from multi-turn conversations → single-turn tweet paraphrasing

**Success criteria**: Optimized steering significantly outperforms Equal-weight steering, demonstrating that Chronicles optimization captures generalizable persona traits, not task-specific patterns.

## Core Principle: Evaluation-Only Usage

### Why LaMP is Evaluation-Only

**The LaMP benchmark serves a fundamentally different role than Conversation Chronicles:**

| Aspect | Conversation Chronicles | LaMP |
|--------|------------------------|------|
| Purpose | Persona optimization (CMA-ES) | Generalization evaluation |
| Profile usage | Training signal for optimization | Judge-only evaluation signal |
| Model input | Conversation history only | Task input only (NO profile) |
| Trait vectors | Optimized via CMA-ES | Pre-trained (frozen) |
| Goal | Learn optimal steering | Test if learned steering generalizes |

### Key Experimental Constraints

1. **NO Persona Optimization**
   - LaMP experiments do NOT run CMA-ES
   - Trait vectors are pre-trained on Conversation Chronicles
   - Vectors are frozen and directly applied to LaMP tasks

2. **Profile Isolation**
   - User profiles are NEVER shown to the generation model
   - Profiles are ONLY used by LLM-as-a-judge during evaluation
   - This tests whether trait vectors capture transferable persona characteristics

3. **Training-Free Steering**
   - No fine-tuning on LaMP data
   - No gradient updates
   - Pure activation steering using pre-computed trait vectors

## Experimental Rationale

### Preventing Judge Overfitting

**Problem**: If we optimize personas on LaMP using the same judge for optimization and evaluation, we risk:
- Judge preference hacking (generating outputs that fool the judge, not genuine persona match)
- Overfitting to judge quirks rather than true persona consistency
- Inflated metrics that don't reflect real generalization

**Solution**: Separate optimization (Chronicles) from evaluation (LaMP)
- Trait vectors are optimized on conversation data (Chronicles)
- Evaluation is performed on single-turn tasks (LaMP)
- Judge never sees optimization process, only final outputs

### Role Separation: Chronicles vs. LaMP

```
┌─────────────────────────────────────────────────────────────┐
│                  CONVERSATION CHRONICLES                     │
│                                                              │
│  Input: Multi-turn conversations                            │
│  Profile: Inferred from conversation style                  │
│  Process: CMA-ES optimization of trait vectors              │
│  Output: Optimized trait steering vectors                   │
│  Judge Role: Optimization signal                            │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       │ Transfer learned vectors
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                        LaMP-7                                │
│                                                              │
│  Input: Single-turn paraphrasing task                       │
│  Profile: 24 historical tweets (judge-only)                 │
│  Process: Apply pre-trained trait vectors (frozen)          │
│  Output: Personalized paraphrases                           │
│  Judge Role: Evaluation-only (compare with profile)         │
└──────────────────────────────────────────────────────────────┘
```

### Why This Design is Critical

1. **Generalization Test**
   - Chronicles: Conversational, multi-turn, implicit personas
   - LaMP: Single-turn, explicit profiles, different domain
   - If vectors transfer → True persona capture, not task overfitting

2. **Judge Reliability**
   - Judge hasn't seen optimization loop on LaMP data
   - Reduces risk of adversarial optimization
   - More trustworthy evaluation metrics

3. **Scientific Validity**
   - Clear separation of train/test
   - Tests hypothesis: "Trait vectors capture transferable persona characteristics"
   - Falsifiable: If no transfer → vectors are task-specific

## Experimental Setup

### Comparison Methods

| Method | Description | Trait Vector Source | Profile in Input |
|--------|-------------|---------------------|------------------|
| **Base** | No persona steering | None | No |
| **Prompt** | Profile-based prompting | None | No (judge sees profile) |
| **Equal** | Random trait vectors | Random initialization | No |
| **Optimized** | Pre-trained trait vectors | Chronicles CMA-ES | No |

### Evaluation Protocol

1. **Generation Phase**
   ```
   Input: Task description (e.g., "Paraphrase this tweet: ...")
   Steering: Apply trait vectors to specified layers
   Output: Generated text
   ```

2. **Evaluation Phase**
   ```
   Judge Input:
   - Generated output
   - User profile (24 historical tweets)
   - Reference output (gold paraphrase)

   Judge Task:
   - Rate persona consistency (1-10)
   - Rate output quality (1-10)
   - Provide rationale
   ```

### Metrics

1. **Primary**
   - Persona Consistency Score (judge-based)
   - Output Quality Score (judge-based)

2. **Secondary**
   - BLEU/ROUGE vs. reference output
   - Perplexity
   - Style transfer metrics (if applicable)

3. **Analysis**
   - Correlation between methods
   - Variance across users
   - Transfer gap (Chronicles → LaMP performance)

## Research Questions

1. **Q1: Do trait vectors generalize across tasks?**
   - Hypothesis: Optimized vectors outperform Equal and Base
   - Test: Compare Optimized vs. Equal on LaMP-7

2. **Q2: Is the improvement due to persona or just better generation?**
   - Hypothesis: Improvement is persona-specific, not general quality
   - Test: Persona Consistency should increase more than Output Quality

3. **Q3: Does profile-based prompting compete with steering?**
   - Hypothesis: Steering is more efficient and consistent
   - Test: Prompt vs. Optimized comparison

4. **Q4: What is the transfer gap?**
   - Hypothesis: Some performance drop vs. Chronicles (domain shift)
   - Test: Compare Chronicles optimization scores vs. LaMP evaluation scores

## Limitations & Future Work

### Current Limitations

1. **LaMP-6 Unavailable**
   - Email subject generation would test formal writing domain
   - Only tweet paraphrasing (informal) available now
   - Consider requesting Avocado access if needed for thesis

2. **Single-Turn Only**
   - LaMP tasks are single-turn
   - Doesn't test multi-turn consistency
   - Chronicles already covers multi-turn

3. **Style vs. Content**
   - Tweet paraphrasing is heavily style-focused
   - May not test content/knowledge personalization
   - Trade-off: Better for style steering validation

### Future Directions

1. **Additional Benchmarks**
   - PersonalLM (if available)
   - Custom multi-turn datasets
   - Domain-specific tasks (technical writing, etc.)

2. **Hybrid Methods**
   - Combine steering + prompting
   - Adaptive steering strength
   - Layer-specific steering strategies

3. **Transfer Learning**
   - Few-shot adaptation on LaMP
   - Online learning during evaluation
   - Meta-learning for faster persona adaptation

## Success Criteria

**Phase 1 is successful if:**

1. ✅ LaMP-7 dataset successfully downloaded and validated
2. ✅ Data structure documented and understood
3. ✅ Experimental assumptions clearly defined
4. ✅ Implementation plan ready for Phase 2
5. ⚠️ LaMP-6 access strategy defined (optional, blocked by Avocado)

**Overall experiment is successful if:**

- Optimized vectors significantly outperform Equal baseline (p < 0.05)
- Persona consistency improves without quality degradation
- Results align with Chronicles findings (validation)
- Clear documentation enables reproducibility
