# Phase 1 API Cost Estimate

**Generated:** 2025-12-09
**Judge Model:** GPT-4o-mini

---

## API Usage Summary

### Judge Model: GPT-4o-mini
- **Provider:** OpenAI
- **Pricing (as of 2024):**
  - Input: $0.150 / 1M tokens
  - Output: $0.600 / 1M tokens

---

## Phase 1 Experiments Breakdown

### 1. Baseline Comparison (Seeds 1, 2, 3)

**Per seed:**
- 7 methods × 20 prompts = 140 evaluations
- Each evaluation = 1 judge call (baseline vs steered comparison)

**Total across 3 seeds:**
- **Total evaluations:** 140 × 3 = **420 judge calls**

**Estimated tokens per judge call:**
- Input tokens:
  - Persona profile: ~500 tokens
  - Prompt: ~50 tokens
  - Baseline response: ~128 tokens
  - Steered response: ~128 tokens
  - Evaluation policy: ~1000 tokens
  - **Total input:** ~1,800 tokens

- Output tokens:
  - Judge reasoning + score: ~200 tokens

**Baseline Comparison Cost:**
```
Input:  420 calls × 1,800 tokens = 756,000 tokens
Output: 420 calls × 200 tokens   = 84,000 tokens

Input cost:  756,000 / 1,000,000 × $0.150 = $0.113
Output cost: 84,000 / 1,000,000 × $0.600  = $0.050

Subtotal: $0.163
```

---

### 2. Cross-Layer Evaluation

**Configuration:**
- 7 methods × 5 layers × 20 prompts = 700 evaluations
- 1 seed

**Cross-Layer Cost:**
```
Input:  700 calls × 1,800 tokens = 1,260,000 tokens
Output: 700 calls × 200 tokens   = 140,000 tokens

Input cost:  1,260,000 / 1,000,000 × $0.150 = $0.189
Output cost: 140,000 / 1,000,000 × $0.600   = $0.084

Subtotal: $0.273
```

---

### 3. Ablation Study

**Configuration:**
- 8 ablation variants × 10 prompts = 80 evaluations
- 1 seed

**Ablation Cost:**
```
Input:  80 calls × 1,800 tokens = 144,000 tokens
Output: 80 calls × 200 tokens   = 16,000 tokens

Input cost:  144,000 / 1,000,000 × $0.150 = $0.022
Output cost: 16,000 / 1,000,000 × $0.600  = $0.010

Subtotal: $0.032
```

---

## Total Phase 1 Cost

| Component | Evaluations | Input Tokens | Output Tokens | Cost |
|-----------|-------------|--------------|---------------|------|
| Baseline (3 seeds) | 420 | 756,000 | 84,000 | $0.163 |
| Cross-Layer | 700 | 1,260,000 | 140,000 | $0.273 |
| Ablation | 80 | 144,000 | 16,000 | $0.032 |
| **TOTAL** | **1,200** | **2,160,000** | **240,000** | **$0.468** |

---

## Cost Summary

### Phase 1 Total: **~$0.47 USD (約70円)**

### Breakdown by Category:
- **Input tokens:** 2.16M tokens → $0.324
- **Output tokens:** 240K tokens → $0.144
- **Total API calls:** 1,200 judge evaluations

---

## Additional Costs (Not API-related)

### GPU Compute (Local)
- Using Meta-Llama-3-8B-Instruct locally
- **Cost:** $0.00 (using local GPU)
- **GPU time:** ~10-15 hours total

### Storage
- Results files: ~50 MB
- Model cache: ~16 GB (already downloaded)
- **Cost:** Negligible

---

## Cost Comparison

### If using GPT-4o instead of GPT-4o-mini:

**GPT-4o pricing:**
- Input: $2.50 / 1M tokens
- Output: $10.00 / 1M tokens

**GPT-4o cost:**
```
Input:  2,160,000 / 1,000,000 × $2.50  = $5.40
Output: 240,000 / 1,000,000 × $10.00   = $2.40

Total: $7.80 USD (約1,170円)
```

**Savings by using GPT-4o-mini:** $7.33 (94% cheaper)

---

## Phase 2 Estimated Cost (Future)

If Phase 2 includes:
- TruthfulQA: 100 samples
- MMLU: 500 samples
- Multi-judge: 20 prompts × 3 judges

**Estimated Phase 2 cost:** ~$0.80 - $1.50 USD

---

## Total Project Cost Estimate

| Phase | Description | Estimated Cost |
|-------|-------------|----------------|
| Phase 1 | Baseline + Cross-layer + Ablation | **$0.47** |
| Phase 2 | TruthfulQA + MMLU + Multi-judge | $0.80 - $1.50 |
| Phase 3 | Extended evaluation | $1.00 - $2.00 |
| **TOTAL** | Full project | **$2.27 - $3.97** |

**Total project cost: 約340-600円**

---

## Cost Optimization Notes

1. **Using GPT-4o-mini:**
   - 94% cheaper than GPT-4o
   - Still maintains good evaluation quality

2. **Local LLM for generation:**
   - Using Llama-3-8B locally (no API cost)
   - Only API cost is for judge model

3. **Batch processing:**
   - Could reduce costs further with OpenAI batch API (50% discount)
   - Not implemented yet

4. **Caching:**
   - Persona profiles and policies could be cached
   - Potential 10-20% reduction in input tokens

---

## Summary

**Phase 1 API cost is extremely low: ~$0.47 USD (約70円)**

This is very affordable for a comprehensive research experiment. The main cost is compute time (10-15 hours on local GPU), not API fees.

---

**Note:** Prices are estimates based on:
- GPT-4o-mini pricing as of December 2024
- Average token counts from similar experiments
- Actual costs may vary by ±20% depending on response lengths
