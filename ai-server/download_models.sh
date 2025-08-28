#!/bin/bash


# NAILA AI Models Download Script
# This script downloads all required AI models


set -e


# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'


# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Please run this script from the ai-server directory${NC}"
    exit 1
fi


# Check if huggingface_hub is available
echo -e "${BLUE}Checking dependencies...${NC}"
if ! uv run python -c "import huggingface_hub" 2>/dev/null; then
    echo -e "${YELLOW}Installing huggingface_hub...${NC}"
    uv add huggingface-hub
fi


# Create model directories
echo -e "${BLUE}Creating model directories...${NC}"
mkdir -p models/{stt,llm,tts,vision}


# Download STT Model (Whisper Small)
echo -e "${BLUE}Downloading Speech-to-Text model (Whisper Small)...${NC}"
echo "Model: ggml-small.en.bin (~244MB)"
uv run python -c "
from huggingface_hub import hf_hub_download
print('Downloading Whisper Small model...')
hf_hub_download(
    repo_id='ggerganov/whisper.cpp',
    filename='ggml-small.en.bin',
    local_dir='models/stt/'
)
print('STT model downloaded successfully')
"


# Download LLM Model (Llama 3.1 8B)
echo -e "${BLUE}Downloading Large Language Model (Llama 3.1 8B Instruct)...${NC}"
echo "Model: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf (~4.9GB)"
echo -e "${YELLOW} This is a large download and may take several minutes...${NC}"
uv run python -c "
from huggingface_hub import hf_hub_download
print('Downloading Llama 3.1 8B Instruct model...')
hf_hub_download(
    repo_id='bartowski/Meta-Llama-3.1-8B-Instruct-GGUF',
    filename='Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
    local_dir='models/llm/'
)
print('LLM model downloaded successfully')
"


# Download TTS Model (Piper)
echo -e "${BLUE}Downloading Text-to-Speech model (Piper - Lessac voice)...${NC}"
echo "Model: en_US-lessac-medium (~17MB)"
uv run python -c "
from huggingface_hub import hf_hub_download
print('Downloading Piper TTS model...')
hf_hub_download(
    repo_id='rhasspy/piper-voices',
    filename='en/en_US/lessac/medium/en_US-lessac-medium.onnx',
    local_dir='models/tts/'
)
hf_hub_download(
    repo_id='rhasspy/piper-voices',
    filename='en/en_US/lessac/medium/en_US-lessac-medium.onnx.json',
    local_dir='models/tts/'
)
print('TTS model downloaded successfully')
"


# Download Vision Model (YOLOv8n)
echo -e "${BLUE}Downloading Computer Vision model (YOLOv8 Nano)...${NC}"
echo "Model: yolov8n.pt (~6MB)"
uv run python -c "
from huggingface_hub import hf_hub_download
print('Downloading YOLOv8n model...')
hf_hub_download(
    repo_id='ultralytics/yolov8',
    filename='yolov8n.pt',
    local_dir='models/vision/'
)
print('Vision model downloaded successfully')
"
fi


echo ""
echo -e "${GREEN}All models downloaded successfully!${NC}"
echo ""
echo "Downloaded models:"
echo "  • STT:    models/stt/ggml-small.en.bin"
echo "  • LLM:    models/llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
echo "  • TTS:    models/tts/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
echo "  • Vision: models/vision/yolov8n.pt"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Check your .env file has the correct model paths"
echo "  2. Run 'uv run python main.py' to start the AI server"