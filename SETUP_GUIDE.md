# Complete Setup Guide

This guide walks you through setting up the Persona Steering framework from scratch.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU with 16GB+ VRAM (recommended)
- 30GB free disk space (for models and data)
- Git

## Step-by-Step Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd persona-steering-cmaes
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

This will install:
- PyTorch with CUDA support
- Transformers library
- CMA-ES optimizer
- OpenAI API client
- All visualization and data processing libraries

**Installation time:** 5-10 minutes depending on your connection.

### 4. Setup HuggingFace Access

#### 4.1. Create HuggingFace Account

1. Go to [huggingface.co](https://huggingface.co) and sign up
2. Verify your email

#### 4.2. Request Llama-3 Access

1. Visit [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
2. Click "Request Access"
3. Accept Meta's license agreement
4. **Wait for approval** (usually instant, can take up to 24 hours)

#### 4.3. Generate Access Token

1. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Name: `persona-steering`
4. Type: **Read**
5. Copy the token (starts with `hf_...`)

#### 4.4. Login via CLI

```bash
# Login interactively
huggingface-cli login

# Paste your token when prompted
# Token will be saved to ~/.cache/huggingface/token
```

**Alternative: Environment Variable**
```bash
# Add to .env file
echo "HF_TOKEN=hf_your_token_here" >> .env

# Or export temporarily
export HF_TOKEN=hf_your_token_here
```

#### 4.5. Verify Access

```bash
python3 << EOF
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
print("✓ Successfully accessed Meta-Llama-3-8B-Instruct")
EOF
```

**First time:** This will download ~16GB to `~/.cache/huggingface/hub/`

### 5. Setup OpenAI API

#### 5.1. Create OpenAI Account

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Add payment method (required for API access)

#### 5.2. Generate API Key

1. Go to [API Keys](https://platform.openai.com/api-keys)
2. Click "+ Create new secret key"
3. Name: `persona-steering`
4. Copy the key (starts with `sk-...`)
5. **Save it securely** - you won't see it again!

#### 5.3. Configure API Key

```bash
# Copy template
cp .env.example .env

# Edit .env
nano .env  # or vim, code, etc.
```

Update `.env`:
```bash
OPENAI_API_KEY=sk-your-actual-key-here
JUDGE_MODEL=gpt-4o-mini  # Recommended for cost efficiency
```

#### 5.4. Verify API Access

```bash
python3 << EOF
import openai
import os
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hi"}],
    max_tokens=5
)
print("✓ OpenAI API is working!")
print(f"Response: {response.choices[0].message.content}")
EOF
```

### 6. Verify Installation

Run the verification script:

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from pathlib import Path

print("Verifying installation...")
print()

# Check packages
print("1. Checking core packages...")
try:
    import torch
    import transformers
    import cma
    import openai
    print(f"   ✓ PyTorch {torch.__version__}")
    print(f"   ✓ Transformers {transformers.__version__}")
    print(f"   ✓ CMA-ES {cma.__version__}")
    print(f"   ✓ OpenAI {openai.__version__}")
except ImportError as e:
    print(f"   ✗ Missing package: {e}")
    sys.exit(1)

print()

# Check modules
print("2. Checking project modules...")
try:
    from persona_opt.cmaes_persona_optimizer import CMAESPersonaOptimizer
    from persona_opt.svd_vector_builder import ActivationCollector
    from persona_judge.persona_profile import generate_persona_profile
    print("   ✓ All core modules importable")
except ImportError as e:
    print(f"   ✗ Module import failed: {e}")
    sys.exit(1)

print()

# Check data
print("3. Checking data files...")
if (Path('data/steering_vectors_v2').exists() and
    len(list(Path('data/steering_vectors_v2').iterdir())) >= 5):
    print("   ✓ Steering vectors found")
else:
    print("   ✗ Steering vectors missing")

if Path('personas').exists() and len(list(Path('personas').iterdir())) >= 10:
    print("   ✓ Personas found")
else:
    print("   ✗ Personas missing")

print()

# Check GPU
print("4. Checking GPU availability...")
if torch.cuda.is_available():
    print(f"   ✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("   ⚠ No GPU detected - inference will be slow")

print()
print("=" * 60)
print("✓ Setup complete! Ready to run experiments.")
print("=" * 60)
EOF
```

### 7. Test Run (Optional)

Try loading the model to ensure everything works:

```bash
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading Llama-3-8B-Instruct...")
print("This may take a few minutes on first run...")

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"✓ Model loaded successfully!")
print(f"✓ Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters")
EOF
```

## Troubleshooting

### Issue: "Access denied to meta-llama/Meta-Llama-3-8B-Instruct"

**Solution:**
1. Check you've requested access on HuggingFace
2. Wait for approval (check email)
3. Verify you're logged in: `huggingface-cli whoami`

### Issue: "CUDA out of memory"

**Solutions:**
1. Reduce batch size in scripts
2. Use `torch.float16` instead of `float32`
3. Use model quantization (4-bit, 8-bit)
4. Use smaller model or CPU inference

### Issue: "OpenAI API rate limit exceeded"

**Solutions:**
1. Add payment method to OpenAI account
2. Wait a few minutes between requests
3. Use `gpt-4o-mini` instead of `gpt-4o` (cheaper, higher limits)

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
# Ensure you're in virtual environment
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

## Next Steps

Once setup is complete:

1. Read the main [README.md](README.md) for usage examples
2. Try the Quick Start tutorial
3. Explore the [docs/](docs/) directory for detailed documentation

## Cost Estimates

### One-time Costs
- Disk space: Free (30GB)
- Model download: Free (bandwidth)

### Per-Use Costs
- GPU compute: Free (if you have GPU) or cloud GPU rental ($0.50-$2/hour)
- OpenAI API:
  - Single persona optimization: $0.50-$2.00
  - Full evaluation (13 personas): $10-$30
  - Uses `gpt-4o-mini` by default (cheapest option)

## Support

If you encounter issues:
1. Check this guide's Troubleshooting section
2. Review error messages carefully
3. Open an issue on GitHub with:
   - Error message
   - Python version
   - GPU info (if applicable)
   - Steps to reproduce

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Taisei Nakata
