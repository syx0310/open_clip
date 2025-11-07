#!/bin/bash

# Complete setup and execution script for DinoV3 + Qwen training
# This script will:
# 1. Install all dependencies
# 2. Run the training script with different modes

set -e  # Exit on error

echo "========================================================================"
echo "DinoV3 + Qwen CLIP Training - Setup and Run"
echo "========================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# Step 1: Check Python
# ============================================================================
echo -e "\n${YELLOW}[Step 1/4] Checking Python installation...${NC}"

if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python version: $PYTHON_VERSION${NC}"

# ============================================================================
# Step 2: Install Dependencies
# ============================================================================
echo -e "\n${YELLOW}[Step 2/4] Installing dependencies...${NC}"

# Check if we should install dependencies
read -p "Install/upgrade dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    echo "Installing Transformers and other dependencies..."
    pip install transformers accelerate pillow

    echo "Installing OpenCLIP in editable mode..."
    cd /home/user/open_clip
    pip install -e .

    echo -e "${GREEN}✓ All dependencies installed${NC}"
else
    echo "Skipping dependency installation"
fi

# ============================================================================
# Step 3: Verify Installation
# ============================================================================
echo -e "\n${YELLOW}[Step 3/4] Verifying installation...${NC}"

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import open_clip; print('OpenCLIP: OK')"

if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo -e "${GREEN}✓ CUDA is available${NC}"
    python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
else
    echo -e "${YELLOW}⚠ CUDA not available, will use CPU (slower)${NC}"
fi

# ============================================================================
# Step 4: Run Training
# ============================================================================
echo -e "\n${YELLOW}[Step 4/4] Running training...${NC}"
echo "Choose training mode:"
echo "1) 1:1 alignment (Standard CLIP) - RECOMMENDED"
echo "2) 1:N alignment (1 text with 2 vision encoders)"
echo "3) N:1 alignment (2 text encoders with 1 vision)"
echo "4) Custom (specify your own arguments)"
echo "5) Skip training"

read -p "Enter choice [1-5]: " choice

cd /home/user/open_clip/examples

case $choice in
    1)
        echo -e "\n${GREEN}Running 1:1 training (DinoV2-Base + Qwen2-0.5B)${NC}"
        python train_dinov3_qwen.py \
            --mode 1to1 \
            --vision-model facebook/dinov2-base \
            --text-model Qwen/Qwen2-0.5B \
            --batch-size 16 \
            --epochs 5 \
            --learning-rate 1e-4 \
            --num-samples 200 \
            --log-interval 5
        ;;

    2)
        echo -e "\n${GREEN}Running 1:N training (1 text + 2 vision encoders)${NC}"
        python train_dinov3_qwen.py \
            --mode 1text_nvision \
            --num-vision 2 \
            --vision-model facebook/dinov2-base \
            --text-model Qwen/Qwen2-0.5B \
            --batch-size 8 \
            --epochs 5 \
            --learning-rate 1e-4 \
            --num-samples 100 \
            --aggregation mean \
            --log-interval 5
        ;;

    3)
        echo -e "\n${GREEN}Running N:1 training (2 text encoders + 1 vision)${NC}"
        python train_dinov3_qwen.py \
            --mode ntext_1vision \
            --num-text 2 \
            --vision-model facebook/dinov2-base \
            --text-model Qwen/Qwen2-0.5B \
            --batch-size 8 \
            --epochs 5 \
            --learning-rate 1e-4 \
            --num-samples 100 \
            --aggregation mean \
            --log-interval 5
        ;;

    4)
        echo -e "\n${GREEN}Enter custom arguments:${NC}"
        read -p "Arguments: " custom_args
        python train_dinov3_qwen.py $custom_args
        ;;

    5)
        echo "Skipping training"
        ;;

    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

# ============================================================================
# Done
# ============================================================================
echo -e "\n${GREEN}========================================================================"
echo "✅ Setup and training complete!"
echo "========================================================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Check the output logs above for training progress"
echo "  2. Checkpoints are saved in: ./checkpoints/"
echo "  3. Replace dummy dataset with your actual data in train_dinov3_qwen.py"
echo ""
echo "To run again:"
echo "  bash examples/setup_and_run.sh"
echo ""
echo "For help:"
echo "  python examples/train_dinov3_qwen.py --help"
echo ""
