# Persona Selection Update (2025-12-15)

## Issue
Original selection included 2 personas that don't exist in Conversation Chronicles dataset:
- episode-31102_A (NOT FOUND)
- episode-229805_A (NOT FOUND)

## Resolution
Replaced with nearby episodes from same numeric ranges, maintaining stratified sampling:

### P5 Replacement (Casual Neutral)
- **Original**: episode-31102_A (target range ~31k)
- **Replacement**: episode-29600_A (Parent, range ~30k)
- **Rationale**: Same numeric range (25k-35k), 44 utterances, Parent relationship provides casual conversational style
- **Stats**: 5 sessions, 44 utts, avg length 90.0, exclamation rate 0.11

### P10 Replacement (Reflective Philosophical)
- **Original**: episode-229805_A (target range ~230k)
- **Replacement**: episode-225888_A (Classmates A, range ~226k)
- **Rationale**: Same numeric range (225k-235k), 41 utterances, late-range episode for philosophical tone
- **Stats**: 5 sessions, 41 utts, avg length 102.8, exclamation rate 0.02

## Final 10 Personas

| ID | Persona ID | Archetype | Episode # | Relationship | Utterances |
|----|------------|-----------|-----------|--------------|------------|
| P1 | episode-184019_A | Formal/Verbose | 184019 | (existing) | ~40 |
| P2 | episode-239427_A | Casual/Emotional | 239427 | (existing) | ~40 |
| P3 | episode-118328_B | Neutral/Concise | 118328 | (existing) | ~40 |
| P4 | episode-5289_A | Formal Emotional | 5289 | Husband and Wife | 40 |
| P5 | episode-29600_A | Casual Neutral | 29600 | Parent and Child | 44 |
| P6 | episode-88279_B | Verbose Intimate | 88279 | Classmates | 42 |
| P7 | episode-132247_A | Concise Detached | 132247 | Neighbors | 40 |
| P8 | episode-166805_A | Conflict Avoidant | 166805 | Neighbors | 42 |
| P9 | episode-196697_B | Advice Centric | 196697 | Classmates | 43 |
| P10 | episode-225888_A | Reflective Philosophical | 225888 | Classmates | 41 |

## Impact on Paper

### Selection Methodology (No Change)
- Stratified numeric sampling still valid
- Replacement maintains same numeric ranges
- Reproducibility preserved (documented substitution)

### Updated Text for Paper
In Experimental Setup section, add footnote:

```latex
\footnote{Two originally selected personas (episode-31102, episode-229805)
were not found in the dataset. They were replaced with nearby episodes
from the same numeric ranges (episode-29600, episode-225888 respectively)
to maintain stratification.}
```

## Extraction Results

All 7 new personas successfully extracted:
- episode-5289_A: ✓ 40 utts
- episode-29600_A: ✓ 44 utts
- episode-88279_B: ✓ 42 utts
- episode-132247_A: ✓ 40 utts
- episode-166805_A: ✓ 42 utts
- episode-196697_B: ✓ 43 utts
- episode-225888_A: ✓ 41 utts

**Status**: Ready for trait extraction and optimization
