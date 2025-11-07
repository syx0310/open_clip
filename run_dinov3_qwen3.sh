#!/bin/bash

# One-click setup and training for DinoV3 + Qwen3-Embedding CLIP
# This script handles installation, verification, and training

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}"
echo "========================================================================"
echo "  DinoV3 + Qwen3-Embedding CLIP Training"
echo "========================================================================"
echo -e "${NC}"
echo ""
echo "Models:"
echo "  Vision: facebook/dinov3-vith16plus-pretrain-lvd1689m (840M params)"
echo "  Text: Qwen/Qwen3-Embedding-4B (4B params)"
echo ""
echo "========================================================================"
echo ""

# ============================================================================
# Step 1: Check Python
# ============================================================================
echo -e "${YELLOW}[Step 1/5] Checking Python...${NC}"

if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python not found!${NC}"
    echo "Please install Python 3.8+ first"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# ============================================================================
# Step 2: Install Dependencies
# ============================================================================
echo -e "\n${YELLOW}[Step 2/5] Installing dependencies...${NC}"

read -p "Install/upgrade dependencies? (y/n) [y]: " -r
REPLY=${REPLY:-y}

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    echo "Installing Transformers (>= 4.56.0 required for DinoV3)..."
    pip install "transformers>=4.56.0" accelerate pillow sentence-transformers

    echo "Installing OpenCLIP..."
    cd /home/user/open_clip
    pip install -e .

    echo -e "${GREEN}✓ All dependencies installed${NC}"
else
    echo "Skipping installation"
fi

# ============================================================================
# Step 3: Verify Installation
# ============================================================================
echo -e "\n${YELLOW}[Step 3/5] Verifying installation...${NC}"

echo -n "Checking PyTorch... "
python -c "import torch; print(f'v{torch.__version__}')" || exit 1

echo -n "Checking Transformers... "
TRANSFORMERS_VERSION=$(python -c "import transformers; print(transformers.__version__)" 2>&1)
echo "$TRANSFORMERS_VERSION"

# Check if transformers version is >= 4.56.0
REQUIRED_VERSION="4.56.0"
if python -c "from packaging import version; import transformers; exit(0 if version.parse(transformers.__version__) >= version.parse('$REQUIRED_VERSION') else 1)" 2>/dev/null; then
    echo -e "${GREEN}✓ Transformers version OK (>= 4.56.0)${NC}"
else
    echo -e "${YELLOW}⚠ Transformers < 4.56.0 detected, upgrading...${NC}"
    pip install --upgrade "transformers>=4.56.0"
fi

echo -n "Checking CUDA... "
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')")
    echo -e "${GREEN}Available - $GPU_NAME ($GPU_MEM)${NC}"
else
    echo -e "${YELLOW}Not available (will use CPU - very slow!)${NC}"
fi

echo -n "Checking OpenCLIP... "
python -c "from open_clip.base_model import TransformersVisionEncoder; print('OK')" || exit 1

# ============================================================================
# Step 4: Choose Configuration
# ============================================================================
echo -e "\n${YELLOW}[Step 4/5] Choose training configuration...${NC}"
echo ""
echo "Select configuration:"
echo "  1) Quick test (small batch, few epochs) - 5 min"
echo "  2) Standard training (recommended) - 1-2 hours"
echo "  3) Full training (high quality) - 3-5 hours"
echo "  4) Custom (specify your own parameters)"
echo "  5) Skip training (just verify installation)"
echo ""

read -p "Enter choice [1-5]: " choice

cd /home/user/open_clip

case $choice in
    1)
        echo -e "\n${GREEN}Running quick test...${NC}"
        python examples/train_dinov3_qwen3.py \
            --batch-size 2 \
            --epochs 2 \
            --num-samples 50 \
            --log-interval 2 \
            --embed-dim 512
        ;;

    2)
        echo -e "\n${GREEN}Running standard training...${NC}"
        python examples/train_dinov3_qwen3.py \
            --batch-size 4 \
            --epochs 10 \
            --num-samples 1000 \
            --log-interval 10 \
            --embed-dim 1024
        ;;

    3)
        echo -e "\n${GREEN}Running full training...${NC}"
        python examples/train_dinov3_qwen3.py \
            --batch-size 8 \
            --epochs 20 \
            --num-samples 5000 \
            --log-interval 20 \
            --embed-dim 1024 \
            --learning-rate 3e-5
        ;;

    4)
        echo -e "\n${GREEN}Custom training${NC}"
        echo "Enter custom arguments:"
        read -p "Arguments: " custom_args
        python examples/train_dinov3_qwen3.py $custom_args
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
# Step 5: Summary
# ============================================================================
echo -e "\n${YELLOW}[Step 5/5] Summary${NC}"

if [ $choice -ne 5 ]; then
    echo ""
    echo "Training completed!"
    echo ""
    echo "Checkpoints saved to: ./checkpoints_dinov3_qwen3/"
    echo ""
    echo "To load a checkpoint:"
    echo "  checkpoint = torch.load('checkpoints_dinov3_qwen3/dinov3_qwen3_epoch_10.pt')"
    echo ""
fi

echo -e "${GREEN}"
echo "========================================================================"
echo "✅ Setup Complete!"
echo "========================================================================"
echo -e "${NC}"
echo ""
echo "Next steps:"
echo "  1. Replace dummy dataset with your real image-text pairs"
echo "  2. Adjust hyperparameters in train_dinov3_qwen3.py"
echo "  3. Run distributed training for multi-GPU"
echo ""
echo "Documentation:"
echo "  - Quick guide: RUN_DINOV3_QWEN3.md"
echo "  - Full docs: REFACTORING_GUIDE.md"
echo ""
echo "To run training again:"
echo "  python examples/train_dinov3_qwen3.py"
echo ""
echo "To see all options:"
echo "  python examples/train_dinov3_qwen3.py --help"
echo ""
