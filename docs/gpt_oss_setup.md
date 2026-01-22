# Local Model Setup Guide

This guide explains how to set up and run local models for use with the persona optimization system.

## Overview

Running models locally allows you to:
- Eliminate API costs during exploration phases
- Run hundreds/thousands of optimization iterations locally
- Reserve GPT-4o for final evaluation only

## Supported Backends

The persona optimization system supports three backends:

1. **HuggingFace** - Direct inference using transformers (recommended)
2. **Ollama** - Via OpenAI-compatible API (alternative)
3. **OpenAI API** - Cloud-based (for comparison/final evaluation)

## Option 1: HuggingFace Backend (Recommended)

### Requirements
- NVIDIA A100 80GB GPU
- Python 3.8+
- PyTorch 2.x with CUDA
- transformers 4.x
- ~16GB disk space for Llama-3.1-8B

### Installation

```bash
# Already installed in your environment
pip list | grep -E "torch|transformers"
# torch 2.4.1
# transformers 4.46.3
```

### Setup HuggingFace Token (for gated models like Llama-3.1)

1. Create a token at https://huggingface.co/settings/tokens
2. Accept the license for Llama-3.1: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
3. Set the token:

```bash
export HF_TOKEN="your_token_here"
huggingface-cli login --token $HF_TOKEN
```

### Usage

```python
from persona_opt.generator import PersonaGenerator
from persona_opt.judge import LLMJudge

# Generator with Llama-3.1-8B
generator = PersonaGenerator(
    model="hf/meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=256,
    temperature=0.7
)

# Judge with Llama-3.1-8B (or use GPT-4o-mini for stability)
judge = LLMJudge(
    model="gpt-4o-mini",  # Recommended: use API for judge
    policy_path="policy/eval_policy_v2.md"
)

# Or use HF for judge too (may be less stable for JSON output)
judge = LLMJudge(
    model="hf/meta-llama/Llama-3.1-8B-Instruct",
    policy_path="policy/eval_policy_v2.md"
)
```

### Run Optimization

```bash
# Generator: Local HF model, Judge: GPT-4o-mini API (recommended)
python3.8 -m persona_opt.run_cma_es \
  --generator_model hf/meta-llama/Llama-3.1-8B-Instruct \
  --judge_model gpt-4o-mini \
  --gens 10 --pop 8 --parents 4

# Both local (more cost-effective, but judge may be less stable)
python3.8 -m persona_opt.run_cma_es \
  --generator_model hf/meta-llama/Llama-3.1-8B-Instruct \
  --judge_model hf/meta-llama/Llama-3.1-8B-Instruct \
  --gens 10 --pop 8 --parents 4
```

---

## Option 2: Ollama Backend (Alternative)

### Requirements

- Ollama installed (`which ollama` should work)
- gpt-oss:20b model pulled (`ollama list`)

### Already Set Up

Your system already has:
- Ollama installed at `/usr/local/bin/ollama`
- Model `gpt-oss:20b` (13 GB) available

###Installation Steps (if needed on another machine)

### 1. Install Ollama

```bash
# Navigate to project root
cd /home/nakata/master_thesis

# Create a new virtual environment with Python 3.12
python3.12 -m venv venv_gpt_oss
source venv_gpt_oss/bin/activate
```

### 2. Install GPT-OSS Package

```bash
# Install with Triton backend for optimal A100 performance
pip install "gpt-oss[triton]"
```

### 3. Download Model Weights

```bash
# Download from Hugging Face (requires huggingface-cli)
pip install huggingface-hub

# Download the original checkpoint
huggingface-cli download openai/gpt-oss-20b \
  --include "original/*" \
  --local-dir /home/nakata/models/gpt-oss-20b
```

## Running the Responses API Server

### Basic Server Startup

```bash
# Activate the environment
source /home/nakata/master_thesis/venv_gpt_oss/bin/activate

# Set CUDA memory allocation strategy (helps with large models)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Start the server
python -m gpt_oss.responses_api.serve \
  --checkpoint /home/nakata/models/gpt-oss-20b/original \
  --inference-backend triton \
  --port 11434
```

The server will start on `http://localhost:11434` and provide an OpenAI Responses API-compatible endpoint.

### Using the Helper Script

For convenience, you can use the provided helper script:

```bash
bash scripts/run_gpt_oss_server.sh
```

## Integrating with Persona Optimization

Once the server is running, you can use it with the persona optimization system:

### Environment Variables

Set these in your shell or `.env` file:

```bash
export GPT_OSS_BASE_URL="http://localhost:11434/v1"
export GPT_OSS_API_KEY="dummy-oss-key"  # Any value works for local server
```

### Using with Generator and Judge

```python
from persona_opt.generator import PersonaGenerator
from persona_opt.judge import LLMJudge

# Initialize with local gpt-oss backend
generator = PersonaGenerator(
    model="gpt-oss-20b-local",
    max_new_tokens=256,
    temperature=0.7
)

judge = LLMJudge(
    model="gpt-oss-20b-local",
    policy_path="policy/eval_policy_v2.md"
)
```

### Running Optimization with Local Model

```bash
# CMA-ES optimization with local gpt-oss
python3.8 -m persona_opt.run_cma_es \
  --gens 10 \
  --pop 8 \
  --parents 4 \
  --generator_model gpt-oss-20b-local \
  --judge_model gpt-oss-20b-local \
  --tau 0.80 \
  --max_new_tokens 256
```

## Smoke Testing

Before running full optimization, test the setup:

```bash
cd /home/nakata/master_thesis/persona2
python3.8 test_gpt_oss_local.py
```

This will:
1. Generate responses using local gpt-oss
2. Evaluate them with local gpt-oss judge
3. Print results to verify everything works

## Troubleshooting

### Out of Memory Errors

If you encounter CUDA OOM errors:
1. Reduce `max_new_tokens` in your generation calls
2. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
3. Ensure no other processes are using the GPU

### Connection Errors

If the client can't connect to the server:
1. Verify the server is running: `curl http://localhost:11434/v1/models`
2. Check `GPT_OSS_BASE_URL` is set correctly
3. Ensure firewall isn't blocking port 11434

### Slow Generation

If generation is slower than expected:
1. Verify you're using the `triton` backend
2. Check GPU utilization: `nvidia-smi`
3. Consider reducing `max_new_tokens`

## Workflow Recommendation

For optimal cost/quality tradeoff:

1. **Exploration Phase**: Use `gpt-oss-20b-local` for both generator and judge
   - Run 20-50 generations to find good trait vectors
   - Cost: ~$0 (only electricity)

2. **Refinement Phase**: Use `gpt-oss-20b-local` generator + `gpt-4o-mini` judge
   - Fine-tune the best candidates with better judge
   - Cost: Moderate

3. **Final Evaluation**: Use `gpt-4o` for both
   - Evaluate top 3-5 candidates only
   - Cost: Minimal

## References

- GPT-OSS GitHub: https://github.com/openai/gpt-oss
- Responses API Spec: OpenAI Harmony Responses API
- Model Card: https://huggingface.co/openai/gpt-oss-20b
