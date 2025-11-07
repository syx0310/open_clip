# 🚀 DinoV3 + Qwen3-Embedding 训练 - 从安装到执行

## ⚡ 最快速的方式 (推荐)

```bash
cd /home/user/open_clip
bash run_dinov3_qwen3.sh
```

就这么简单！脚本会自动处理一切。

---

## 📦 手动安装和运行

### 第1步: 安装依赖 (约5分钟)

```bash
# 进入项目目录
cd /home/user/open_clip

# 安装 PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装 Transformers (DinoV3需要>=4.56.0)
pip install "transformers>=4.56.0" accelerate pillow sentence-transformers

# 安装 OpenCLIP
pip install -e .
```

### 第2步: 验证安装

```bash
# 检查PyTorch
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"

# 检查Transformers版本 (必须>=4.56.0)
python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"

# 检查CUDA
python -c "import torch; print(f'✓ CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# 检查OpenCLIP
python -c "from open_clip.base_model import TransformersVisionEncoder; print('✓ OpenCLIP OK')"
```

### 第3步: 运行训练

```bash
# 开始训练！
python examples/train_dinov3_qwen3.py
```

**就这样！训练开始了！** 🎉

---

## 🎯 使用的模型

### DinoV3 ViT-H+ (Vision)
- **HuggingFace**: `facebook/dinov3-vith16plus-pretrain-lvd1689m`
- **参数**: 840M
- **特点**:
  - 最先进的视觉基础模型
  - 在17亿图像上预训练
  - 无需微调即可用于多种视觉任务

### Qwen3-Embedding-4B (Text)
- **HuggingFace**: `Qwen/Qwen3-Embedding-4B`
- **参数**: 4B
- **特点**:
  - 支持100+语言
  - 32K上下文长度
  - 嵌入维度可调 (32-2560)
  - MTEB排行榜领先

---

## ⚙️ 配置选项

### 快速测试 (5分钟)

```bash
python examples/train_dinov3_qwen3.py \
    --batch-size 2 \
    --epochs 2 \
    --num-samples 50 \
    --embed-dim 512
```

### 标准训练 (1-2小时)

```bash
python examples/train_dinov3_qwen3.py \
    --batch-size 4 \
    --epochs 10 \
    --num-samples 1000 \
    --embed-dim 1024
```

### 完整训练 (3-5小时)

```bash
python examples/train_dinov3_qwen3.py \
    --batch-size 8 \
    --epochs 20 \
    --num-samples 5000 \
    --embed-dim 1024 \
    --learning-rate 3e-5
```

---

## 💾 GPU内存需求

| GPU | 批量大小 | 嵌入维度 | 状态 |
|-----|---------|---------|------|
| 24GB (RTX 3090/4090) | 2 | 512 | ✅ 可用 |
| 40GB (A100) | 4-8 | 1024 | ✅ 推荐 |
| 80GB (A100) | 16 | 2048 | ✅ 最佳 |

如果遇到内存不足:
```bash
# 减小批量大小
python examples/train_dinov3_qwen3.py --batch-size 2

# 或减小嵌入维度
python examples/train_dinov3_qwen3.py --embed-dim 512
```

---

## 📊 训练输出预览

```
================================================================================
DinoV3 + Qwen3-Embedding CLIP Training
================================================================================
Vision: facebook/dinov3-vith16plus-pretrain-lvd1689m
Text: Qwen/Qwen3-Embedding-4B
Embed dim: 1024
Device: cuda
Batch size: 4
================================================================================

[1/5] Creating model...
Loading DinoV3 vision encoder...
✓ DinoV3 vision encoder loaded
Loading Qwen3-Embedding text encoder...
✓ Qwen3-Embedding text encoder loaded
✓ Model created successfully
  Total parameters: 4,867,234,816
  Trainable parameters: 4,867,234,816

[2/5] Creating dataset...
✓ Dataset: 1000 samples, 250 batches

[3/5] Creating optimizer...
✓ Optimizer: AdamW, Total steps: 2500

[4/5] Training...
--------------------------------------------------------------------------------
Epoch  1 [  0/250] Loss: 6.9078 Scale: 14.27 LR: 0.000050
Epoch  1 [ 10/250] Loss: 6.4521 Scale: 14.45 LR: 0.000050
...
✓ Epoch 10/10 | Avg Loss: 2.3456
--------------------------------------------------------------------------------

[5/5] Saving final model...
✓ Checkpoint saved: ./checkpoints_dinov3_qwen3/dinov3_qwen3_epoch_10.pt

✅ TRAINING COMPLETED!
```

---

## 🔧 所有可用参数

```bash
python examples/train_dinov3_qwen3.py --help
```

主要参数:
- `--vision-model`: DinoV3模型名称 (默认: vith16plus)
- `--text-model`: Qwen3模型名称 (默认: 4B)
- `--embed-dim`: 嵌入维度 (32-2560, 默认: 1024)
- `--batch-size`: 批量大小 (默认: 8)
- `--epochs`: 训练轮数 (默认: 10)
- `--learning-rate`: 学习率 (默认: 5e-5)
- `--device`: 设备 (cuda/cpu, 默认: cuda)

---

## 🎨 使用其他DinoV3变体

### DinoV3-Small (最小, 22M参数)
```bash
python examples/train_dinov3_qwen3.py \
    --vision-model facebook/dinov3-vits16-pretrain-lvd1689m \
    --batch-size 32
```

### DinoV3-Base (平衡, 86M参数)
```bash
python examples/train_dinov3_qwen3.py \
    --vision-model facebook/dinov3-vitb16-pretrain-lvd1689m \
    --batch-size 16
```

### DinoV3-Large (高性能, 304M参数)
```bash
python examples/train_dinov3_qwen3.py \
    --vision-model facebook/dinov3-vitl16-pretrain-lvd1689m \
    --batch-size 8
```

### DinoV3-H+ (最佳, 840M参数) - 默认
```bash
python examples/train_dinov3_qwen3.py \
    --vision-model facebook/dinov3-vith16plus-pretrain-lvd1689m \
    --batch-size 4
```

---

## 🐛 常见问题

### ❌ 问题: Transformers版本太旧

```bash
# 错误信息: DinoV3 requires transformers >= 4.56.0

# 解决:
pip install --upgrade "transformers>=4.56.0"
```

### ❌ 问题: CUDA Out of Memory

```bash
# 解决方案1: 减小批量
python examples/train_dinov3_qwen3.py --batch-size 2

# 解决方案2: 减小嵌入维度
python examples/train_dinov3_qwen3.py --embed-dim 512

# 解决方案3: 使用更小的DinoV3
python examples/train_dinov3_qwen3.py \
    --vision-model facebook/dinov3-vitb16-pretrain-lvd1689m
```

### ❌ 问题: 模型下载慢

```bash
# 使用HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com
python examples/train_dinov3_qwen3.py
```

---

## 📚 文档

- **快速开始**: [START_HERE.md](START_HERE.md) (本文档)
- **详细指南**: [RUN_DINOV3_QWEN3.md](RUN_DINOV3_QWEN3.md)
- **完整API**: [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md)
- **所有命令**: [COMMANDS.md](COMMANDS.md)

---

## ✅ 检查清单

在开始之前确保:

- [ ] Python >= 3.8
- [ ] PyTorch >= 2.0
- [ ] Transformers >= 4.56.0 (重要!)
- [ ] CUDA可用 (推荐)
- [ ] GPU内存 >= 24GB
- [ ] `pip install -e .` 已执行

---

## 🎉 开始训练！

**方式1: 一键运行**
```bash
bash run_dinov3_qwen3.sh
```

**方式2: 手动运行**
```bash
python examples/train_dinov3_qwen3.py
```

**方式3: 自定义配置**
```bash
python examples/train_dinov3_qwen3.py \
    --batch-size 4 \
    --embed-dim 1024 \
    --epochs 10
```

---

## 📧 需要帮助?

查看 [RUN_DINOV3_QWEN3.md](RUN_DINOV3_QWEN3.md) 获取:
- 完整参数说明
- 内存优化技巧
- 替换真实数据的方法
- 高级功能 (1:N对齐)
- 故障排除详细指南

---

**现在开始训练你的DinoV3 + Qwen3 CLIP模型吧！** 🚀
