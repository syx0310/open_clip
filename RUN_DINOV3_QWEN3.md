# DinoV3 + Qwen3-Embedding 训练指南

## 模型信息

### DinoV3 ViT-H+
- **模型名称**: `facebook/dinov3-vith16plus-pretrain-lvd1689m`
- **参数量**: 840M
- **架构**: Vision Transformer (ViT-H+, patch size 16)
- **预训练数据**: LVD-1689M (17亿图像)
- **性能**: ImageNet-ReaL 90.3, Oxford-5k 64.5

### Qwen3-Embedding-4B
- **模型名称**: `Qwen/Qwen3-Embedding-4B`
- **参数量**: 4B
- **上下文长度**: 32K
- **嵌入维度**: 32-2560 (可配置)
- **语言支持**: 100+ 语言
- **性能**: MTEB多语言 69.45, 英语 74.60

---

## 🚀 快速开始

### 步骤 1: 安装依赖

```bash
cd /home/user/open_clip

# 安装 PyTorch (根据你的CUDA版本)
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装 Transformers 和其他依赖
pip install transformers>=4.56.0 accelerate pillow sentence-transformers

# 安装 OpenCLIP
pip install -e .
```

### 步骤 2: 验证安装

```bash
# 检查 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 检查 Transformers (需要 >= 4.56.0 for DinoV3)
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 检查 CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 检查 OpenCLIP
python -c "from open_clip.base_model import TransformersVisionEncoder; print('OpenCLIP: OK')"
```

### 步骤 3: 运行训练

#### 选项 A: 使用默认配置（推荐首次测试）

```bash
python examples/train_dinov3_qwen3.py
```

#### 选项 B: 自定义配置

```bash
python examples/train_dinov3_qwen3.py \
    --batch-size 4 \
    --embed-dim 1024 \
    --epochs 10 \
    --learning-rate 5e-5 \
    --device cuda
```

---

## 💾 内存需求和推荐配置

| 配置 | GPU 内存 | 批量大小 | 嵌入维度 | 预计训练时间 |
|------|----------|---------|---------|-------------|
| **最小配置** | 24GB | 2 | 512 | ~3小时 |
| **推荐配置** | 40GB | 4-8 | 1024 | ~1.5小时 |
| **高性能配置** | 80GB | 16 | 2048 | ~40分钟 |

**注意**: DinoV3 ViT-H+ (840M) + Qwen3-4B 总共约 5B 参数，需要较大显存。

---

## ⚙️ 完整参数说明

```bash
python examples/train_dinov3_qwen3.py \
    --vision-model facebook/dinov3-vith16plus-pretrain-lvd1689m  # DinoV3模型
    --text-model Qwen/Qwen3-Embedding-4B                         # Qwen3模型
    --embed-dim 1024                                             # 嵌入维度 (32-2560)
    --batch-size 4                                               # 批量大小
    --epochs 10                                                  # 训练轮数
    --learning-rate 5e-5                                         # 学习率
    --weight-decay 0.01                                          # 权重衰减
    --max-grad-norm 1.0                                          # 梯度裁剪
    --warmup-steps 500                                           # 预热步数
    --num-samples 1000                                           # 样本数(dummy)
    --image-size 224                                             # 图像大小
    --num-workers 4                                              # 数据加载线程
    --device cuda                                                # 设备
    --checkpoint-dir ./checkpoints_dinov3_qwen3                  # 检查点目录
    --log-interval 10                                            # 日志间隔
    --save-interval 5                                            # 保存间隔
```

---

## 🔥 使用其他 DinoV3 变体

DinoV3 有多个大小的模型可选：

```bash
# ViT-Small (22M 参数) - 最小配置
python examples/train_dinov3_qwen3.py \
    --vision-model facebook/dinov3-vits16-pretrain-lvd1689m \
    --batch-size 32 \
    --embed-dim 512

# ViT-Base (86M 参数) - 平衡配置
python examples/train_dinov3_qwen3.py \
    --vision-model facebook/dinov3-vitb16-pretrain-lvd1689m \
    --batch-size 16 \
    --embed-dim 768

# ViT-Large (304M 参数) - 高性能配置
python examples/train_dinov3_qwen3.py \
    --vision-model facebook/dinov3-vitl16-pretrain-lvd1689m \
    --batch-size 8 \
    --embed-dim 1024

# ViT-H+ (840M 参数) - 默认/最佳配置
python examples/train_dinov3_qwen3.py \
    --vision-model facebook/dinov3-vith16plus-pretrain-lvd1689m \
    --batch-size 4 \
    --embed-dim 1024

# ConvNeXt-Small - 使用卷积网络
python examples/train_dinov3_qwen3.py \
    --vision-model facebook/dinov3-convnext-small-pretrain-lvd1689m \
    --batch-size 16 \
    --embed-dim 768
```

---

## 🐛 故障排除

### 1. 内存不足 (CUDA Out of Memory)

```bash
# 方法1: 减小批量大小
python examples/train_dinov3_qwen3.py --batch-size 2

# 方法2: 减小嵌入维度
python examples/train_dinov3_qwen3.py --embed-dim 512

# 方法3: 使用更小的DinoV3模型
python examples/train_dinov3_qwen3.py \
    --vision-model facebook/dinov3-vitb16-pretrain-lvd1689m \
    --batch-size 16

# 方法4: 启用梯度检查点（如果实现了）
# python examples/train_dinov3_qwen3.py --gradient-checkpointing
```

### 2. Transformers 版本不兼容

```bash
# DinoV3 需要 transformers >= 4.56.0
pip install --upgrade transformers

# 验证版本
python -c "import transformers; print(transformers.__version__)"
```

### 3. 模型下载失败

```bash
# 使用 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型
huggingface-cli download facebook/dinov3-vith16plus-pretrain-lvd1689m
huggingface-cli download Qwen/Qwen3-Embedding-4B
```

### 4. CUDA 不可用

```bash
# 使用 CPU (非常慢，仅测试用)
python examples/train_dinov3_qwen3.py \
    --device cpu \
    --batch-size 1 \
    --num-samples 10
```

---

## 📊 训练输出示例

```
================================================================================
DinoV3 + Qwen3-Embedding CLIP Training
================================================================================
Vision: facebook/dinov3-vith16plus-pretrain-lvd1689m
Text: Qwen/Qwen3-Embedding-4B
Embed dim: 1024
Device: cuda
Batch size: 4
Learning rate: 5e-05
================================================================================

[1/5] Creating model...
================================================================================
Creating DinoV3 + Qwen3-Embedding CLIP Model
================================================================================
Loading DinoV3 vision encoder: facebook/dinov3-vith16plus-pretrain-lvd1689m
  Architecture: ViT-H+ with 840M parameters
  Input size: 224x224
✓ DinoV3 vision encoder loaded
Loading Qwen3-Embedding text encoder: Qwen/Qwen3-Embedding-4B
  Architecture: 4B parameters, 32K context
  Max embedding dim: 2560
✓ Qwen3-Embedding text encoder loaded
================================================================================
✓ Model created successfully
  Total parameters: 4,867,234,816
  Trainable parameters: 4,867,234,816
  Embedding dimension: 1024
================================================================================

[2/5] Creating dataset...
⚠️  Using dummy dataset! Replace with your actual data.
✓ Dataset: 1000 samples, 250 batches

[3/5] Creating optimizer...
✓ Optimizer: AdamW, Total steps: 2500

[4/5] Training...
--------------------------------------------------------------------------------
Epoch  1 [  0/250] Loss: 6.9078 Scale: 14.27 LR: 0.000050
Epoch  1 [ 10/250] Loss: 6.4521 Scale: 14.45 LR: 0.000050
...
✓ Epoch 1/10 | Avg Loss: 5.2134
...
✓ Checkpoint saved: ./checkpoints_dinov3_qwen3/dinov3_qwen3_epoch_5.pt
...
✓ Epoch 10/10 | Avg Loss: 2.3456
--------------------------------------------------------------------------------

[5/5] Saving final model...
✓ Checkpoint saved: ./checkpoints_dinov3_qwen3/dinov3_qwen3_epoch_10.pt

[Test] Running inference test...
✓ Image features: torch.Size([4, 1024])
✓ Text features: torch.Size([4, 1024])
✓ Similarity matrix (diagonal):
    Pair 0: 0.7823
    Pair 1: 0.8012
    Pair 2: 0.7654
    Pair 3: 0.7891

================================================================================
✅ TRAINING COMPLETED!
Checkpoints: ./checkpoints_dinov3_qwen3
================================================================================
```

---

## 📝 替换为真实数据

编辑 `train_dinov3_qwen3.py` 中的 `DummyImageTextDataset`:

```python
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

class RealImageTextDataset(Dataset):
    def __init__(self, image_paths, texts):
        self.image_paths = image_paths
        self.texts = texts

        # 加载DinoV3的图像处理器
        self.image_processor = AutoImageProcessor.from_pretrained(
            'facebook/dinov3-vith16plus-pretrain-lvd1689m'
        )

        # 加载Qwen3的tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            'Qwen/Qwen3-Embedding-4B',
            trust_remote_code=True
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载并处理图像
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.image_processor(image, return_tensors='pt')['pixel_values'][0]

        # 标记化文本
        text = self.tokenizer(
            self.texts[idx],
            max_length=512,  # 或更长，Qwen3支持32K
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )['input_ids'][0]

        return image, text
```

---

## 🔬 高级功能

### 1:N 对齐（1个文本 + 多个DinoV3变体）

```python
# 修改脚本以使用MultiVisionCLIP
from open_clip.multi_encoder_model import MultiVisionCLIP

vision_encoders = [
    TransformersVisionEncoder('facebook/dinov3-vitb16-pretrain-lvd1689m', 1024),
    TransformersVisionEncoder('facebook/dinov3-vitl16-pretrain-lvd1689m', 1024),
    TransformersVisionEncoder('facebook/dinov3-vith16plus-pretrain-lvd1689m', 1024),
]

model = MultiVisionCLIP(
    vision_encoders=vision_encoders,
    text_encoder=text_encoder,
    embed_dim=1024
)
```

### N:1 对齐（多个文本编码器 + 1个DinoV3）

```python
from open_clip.multi_encoder_model import MultiTextCLIP

text_encoders = [
    TransformersTextEncoder('Qwen/Qwen3-Embedding-4B', 1024),
    TransformersTextEncoder('sentence-transformers/all-mpnet-base-v2', 1024),
]

model = MultiTextCLIP(
    vision_encoder=vision_encoder,
    text_encoders=text_encoders,
    embed_dim=1024
)
```

---

## 📚 参考资料

- **DinoV3 论文**: https://arxiv.org/abs/2508.10104
- **DinoV3 GitHub**: https://github.com/facebookresearch/dinov3
- **DinoV3 HuggingFace**: https://huggingface.co/collections/facebook/dinov3-677130166a11dd8763cba916
- **Qwen3-Embedding**: https://huggingface.co/Qwen/Qwen3-Embedding-4B
- **HuggingFace Transformers Docs**: https://huggingface.co/docs/transformers/model_doc/dinov3

---

## ✅ 检查清单

- [ ] Python >= 3.8
- [ ] PyTorch >= 2.0 已安装
- [ ] Transformers >= 4.56.0 已安装
- [ ] CUDA 可用 (推荐)
- [ ] GPU 内存 >= 24GB (最低)
- [ ] 已克隆 open_clip 仓库
- [ ] 已安装 `pip install -e .`
- [ ] 准备好图像-文本数据集（或使用dummy数据测试）

---

**准备好了吗？运行下面的命令开始训练！**

```bash
python examples/train_dinov3_qwen3.py
```
