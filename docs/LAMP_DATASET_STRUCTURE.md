# LaMP Dataset Structure Summary

## LaMP-6 Structure Summary (Email Subject Generation)

### Data Availability
**⚠️ CRITICAL ISSUE**: LaMP-6 requires the Avocado Research Email Collection, which is **NOT publicly available**.
- Dataset must be obtained through Linguistic Data Consortium (LDC): https://catalog.ldc.upenn.edu/LDC2015T03
- Requires signing two license agreements
- LaMP provides only sample IDs and code to generate the dataset after obtaining Avocado access

### Dataset Statistics
- Train samples: 4,840
- Dev samples: 1,353
- Test samples: 1,246
- Profile entries per sample: 14

### Data Structure
```json
{
  "id": "500",
  "input": "147-000874-EM.txt",  // Reference to email text file (NOT included)
  "profile": [
    {
      "text": "147-000137-EM.txt",  // Reference to profile email (NOT included)
      "id": "5000"
    },
    // ... 13 more profile entries
  ]
}
```

**Gold Outputs Format:**
```json
{
  "task": "LaMP_6",
  "golds": [
    {
      "id": "500",
      "output": "147-000874-EM.txt"  // Reference to target subject file (NOT included)
    }
  ]
}
```

### Notes
- **input**: Reference filename to email body (actual text NOT downloaded)
- **profile**: List of 14 reference filenames to user's previous emails (actual text NOT downloaded)
- **target**: Reference filename to target email subject (actual text NOT downloaded)
- **Single-turn**: Yes - each sample is a standalone subject generation task
- **BLOCKER**: Cannot proceed with LaMP-6 experiments without Avocado dataset access

---

## LaMP-7 Structure Summary (Tweet Paraphrasing)

### Data Availability
✅ **FULLY ACCESSIBLE** - All text content is included in the JSON files

### Dataset Statistics
- Train samples: 10,437
- Dev samples: 1,500
- Test samples: 1,496
- Profile entries per sample: 24
- Average profile entry length: ~92 chars
- Input prompt length: ~131 chars

### Data Structure
```json
{
  "id": "600",
  "input": "Paraphrase the following tweet without any explanation before or after it: I'm currently enjoying the album \"Listen to Eason Chan.\"",
  "profile": [
    {
      "text": "SARS .. H1N1 .. Air France ..  please cherish your life, people ..",
      "id": "6000"
    },
    {
      "text": "\"See ... You make the world go weird ...\" from weiwei's SMS ",
      "id": "6001"
    },
    // ... 22 more profile entries (user's historical tweets)
  ]
}
```

**Gold Outputs Format:**
```json
{
  "task": "LaMP_7",
  "golds": [
    {
      "id": "600",
      "output": "Listening to \"Listen to Eason Chan\" it's a good album "
    }
  ]
}
```

### Example Profile Entry
```
Profile ID: 6003
Text: "listening to eason's 2006 album .. What's going on...? This is my favourite eason album  it's 3.38am"
```
This shows the user's writing style characteristics:
- Casual tone with ellipses (..)
- Stream-of-consciousness style
- Time/context mentions
- Informal capitalization

### Notes
- **input**: Full text prompt asking to paraphrase a given tweet
- **profile**: List of 24 actual historical tweets from the user (full text included)
- **target**: The expected paraphrased output in the user's style
- **Single-turn**: Yes - each sample is a standalone paraphrasing task
- **Profile utility**: Rich signal for persona/style adaptation (casual language, emoji usage, topic preferences, temporal patterns)

---

## Experimental Design Implications

### LaMP-6: BLOCKED
Cannot use LaMP-6 without obtaining Avocado dataset access from LDC. This requires institutional approval and license agreements.

**Recommendation**: Focus exclusively on LaMP-7 for Phase 1 evaluation.

### LaMP-7: READY
- Full dataset available
- Rich user profiles with 24 historical tweets
- Clear style variation signals (tone, formality, emoji usage, etc.)
- Suitable for training-free persona steering evaluation

### Design Concerns

1. **Judge Overfitting Risk**
   - LaMP-7 profiles are highly distinctive (casual tweets vs. formal text)
   - LLM judges may rely on surface-level style matching rather than semantic quality
   - Mitigation: Include both style-aware and style-agnostic evaluation metrics

2. **Profile Length**
   - 24 tweets × ~92 chars = ~2,200 chars per profile
   - May exceed context limits if used naively in prompts
   - Design: Profiles should ONLY be used in judge evaluation, NOT in generation input

3. **Task Simplicity**
   - Tweet paraphrasing is relatively constrained
   - May not fully test persona steering capabilities compared to open-ended generation
   - Trade-off: Better for controlled evaluation, less realistic than multi-turn conversations

4. **Baseline Comparison**
   - Need to define:
     - **Base**: No persona steering
     - **Prompt**: Profile-based prompting (for judge only)
     - **Equal**: Random trait vectors
     - **Optimized**: Pre-trained trait vectors from Conversation Chronicles

5. **Evaluation Protocol**
   - User profile should NOT be shown to generation model
   - User profile SHOULD be shown to judge for persona consistency evaluation
   - Prevents "profile memorization" and tests true generalization of trait vectors
