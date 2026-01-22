#!/bin/bash
#
# GPT-OSS-20B Server Startup Script
#
# This script starts the gpt-oss-20b Responses API server for local inference.
# It assumes you have already installed gpt-oss and downloaded the model weights.
#
# Usage:
#   bash scripts/run_gpt_oss_server.sh
#
# Prerequisites:
#   1. Python 3.12+ environment with gpt-oss installed
#   2. Model weights downloaded to /home/nakata/models/gpt-oss-20b/original
#   3. NVIDIA A100 80GB or similar GPU
#
# The server will run on http://localhost:11434 by default.
#

set -e  # Exit on error

# Configuration
VENV_PATH="/home/nakata/master_thesis/venv_gpt_oss"
MODEL_PATH="/home/nakata/models/gpt-oss-20b/original"
INFERENCE_BACKEND="triton"
PORT="11434"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  GPT-OSS-20B Server Startup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
    echo ""
    echo "Please create the environment first:"
    echo "  cd /home/nakata/master_thesis"
    echo "  python3.12 -m venv venv_gpt_oss"
    echo "  source venv_gpt_oss/bin/activate"
    echo "  pip install 'gpt-oss[triton]'"
    exit 1
fi

# Check if model weights exist
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model weights not found at $MODEL_PATH${NC}"
    echo ""
    echo "Please download the model first:"
    echo "  pip install huggingface-hub"
    echo "  huggingface-cli download openai/gpt-oss-20b \\"
    echo "    --include 'original/*' \\"
    echo "    --local-dir /home/nakata/models/gpt-oss-20b"
    exit 1
fi

# Check if GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: nvidia-smi not found. GPU may not be available.${NC}"
else
    echo -e "${GREEN}GPU Status:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source "$VENV_PATH/bin/activate"

# Set CUDA memory allocation strategy
echo -e "${GREEN}Setting CUDA memory configuration...${NC}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print configuration
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Model Path: $MODEL_PATH"
echo "  Inference Backend: $INFERENCE_BACKEND"
echo "  Port: $PORT"
echo "  Base URL: http://localhost:$PORT/v1"
echo ""

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}Warning: Port $PORT is already in use.${NC}"
    echo "If you want to restart the server, first kill the existing process:"
    echo "  lsof -ti:$PORT | xargs kill -9"
    echo ""
    read -p "Press Enter to continue anyway or Ctrl+C to abort..."
fi

# Start the server
echo -e "${GREEN}Starting GPT-OSS-20B Responses API server...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

python -m gpt_oss.responses_api.serve \
  --checkpoint "$MODEL_PATH" \
  --inference-backend "$INFERENCE_BACKEND" \
  --port "$PORT"

# This line will only be reached if the server exits
echo ""
echo -e "${GREEN}Server stopped.${NC}"
