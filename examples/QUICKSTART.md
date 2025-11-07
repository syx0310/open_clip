# Quick Start: DinoV3 + Qwen Training

## 🚀 一键运行

```bash
cd /home/user/open_clip
bash examples/setup_and_run.sh
```

这个脚本会自动：
1. 检查Python环境
2. 安装所有依赖
3. 提供交互式菜单选择训练模式

---

## 📋 手动安装和运行

### 步骤1: 安装依赖

```bash
# 进入项目目录
cd /home/user/open_clip

# 安装PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装Transformers和其他依赖
pip install transformers accelerate pillow

# 安装OpenCLIP
pip install -e .
```

### 步骤2: 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 步骤3: 运行训练

#### 选项A: 1:1 训练 (标准CLIP) ⭐ 推荐

```bash
python examples/train_dinov3_qwen.py \
    --mode 1to1 \
    --vision-model facebook/dinov2-base \
    --text-model Qwen/Qwen2-0.5B \
    --batch-size 16 \
    --epochs 10 \
    --learning-rate 1e-4
```

#### 选项B: 1:N 训练 (1个文本编码器 + N个视觉编码器)

```bash
python examples/train_dinov3_qwen.py \
    --mode 1text_nvision \
    --num-vision 3 \
    --vision-model facebook/dinov2-base \
    --text-model Qwen/Qwen2-0.5B \
    --batch-size 8 \
    --epochs 10 \
    --aggregation mean
```

#### 选项C: N:1 训练 (N个文本编码器 + 1个视觉编码器)

```bash
python examples/train_dinov3_qwen.py \
    --mode ntext_1vision \
    --num-text 2 \
    --vision-model facebook/dinov2-base \
    --text-model Qwen/Qwen2-0.5B \
    --batch-size 8 \
    --epochs 10 \
    --aggregation mean
```

---

## 🎯 使用真实的DinoV3和Qwen模型

### DinoV3视觉模型

```bash
# 使用DinoV3-ViT-H/16+ (推荐用于生产)
python examples/train_dinov3_qwen.py \
    --vision-model facebook/dinov2-giant \
    --text-model Qwen/Qwen2-7B \
    --embed-dim 1536 \
    --batch-size 4
```

可用的DinoV2/V3模型:
- `facebook/dinov2-small` - 22M参数
- `facebook/dinov2-base` - 86M参数 (推荐测试用)
- `facebook/dinov2-large` - 304M参数
- `facebook/dinov2-giant` - 1.1B参数 (最强性能)

### Qwen文本模型

```bash
# 使用Qwen2-7B或Qwen3-Embedding-4B
python examples/train_dinov3_qwen.py \
    --text-model Qwen/Qwen2-7B \
    --embed-dim 2048
```

可用的Qwen模型:
- `Qwen/Qwen2-0.5B` - 0.5B参数 (推荐测试用)
- `Qwen/Qwen2-1.5B` - 1.5B参数
- `Qwen/Qwen2-7B` - 7B参数 (推荐生产用)

---

## 📊 内存需求和批量大小建议

| 配置 | GPU内存 | 批量大小 | 训练时间(1000步) |
|------|---------|---------|------------------|
| DinoV2-Base + Qwen2-0.5B | 8GB | 32 | ~20分钟 |
| DinoV2-Large + Qwen2-1.5B | 16GB | 16 | ~40分钟 |
| DinoV2-Giant + Qwen2-7B | 40GB | 4 | ~2小时 |

---

## 🔧 常见参数说明

```bash
python examples/train_dinov3_qwen.py \
    --mode 1to1                          # 训练模式: 1to1, 1text_nvision, ntext_1vision
    --vision-model facebook/dinov2-base  # HuggingFace视觉模型
    --text-model Qwen/Qwen2-0.5B        # HuggingFace文本模型
    --embed-dim 768                      # 嵌入维度
    --batch-size 16                      # 批量大小
    --epochs 10                          # 训练轮数
    --learning-rate 1e-4                 # 学习率
    --num-samples 1000                   # 样本数(仅dummy数据)
    --device cuda                        # 设备: cuda或cpu
    --checkpoint-dir ./checkpoints       # 检查点保存目录
    --log-interval 10                    # 日志输出间隔
```

查看所有参数:
```bash
python examples/train_dinov3_qwen.py --help
```

---

## 🐛 故障排除

### 内存不足 (Out of Memory)

```bash
# 减小批量大小
python examples/train_dinov3_qwen.py --batch-size 8

# 使用更小的模型
python examples/train_dinov3_qwen.py \
    --vision-model facebook/dinov2-small \
    --text-model Qwen/Qwen2-0.5B
```

### CUDA不可用

```bash
# 使用CPU (较慢)
python examples/train_dinov3_qwen.py --device cpu --batch-size 4
```

### 模型下载失败

```bash
# 设置HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型到本地
python examples/train_dinov3_qwen.py \
    --vision-model /path/to/local/dinov2 \
    --text-model /path/to/local/qwen
```

---

## 📝 替换为真实数据

编辑 `train_dinov3_qwen.py` 中的 `DummyImageTextDataset` 类:

```python
class YourImageTextDataset(Dataset):
    def __init__(self, image_dir, text_file, tokenizer, image_processor):
        self.images = load_image_paths(image_dir)
        self.texts = load_texts(text_file)
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __getitem__(self, idx):
        # 加载图像
        image = Image.open(self.images[idx]).convert('RGB')
        image = self.image_processor(image, return_tensors='pt')['pixel_values'][0]

        # 标记化文本
        text = self.tokenizer(
            self.texts[idx],
            max_length=77,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )['input_ids'][0]

        return image, text
```

---

## 🎓 训练输出示例

```
========================================================================
CLIP Training: DinoV3 + Qwen Embedding
========================================================================
Mode: 1to1
Vision model: facebook/dinov2-base
Text model: Qwen/Qwen2-0.5B
Embedding dimension: 768
Device: cuda
Batch size: 16
Learning rate: 0.0001
========================================================================

[1/5] Creating model...
✓ Model created with 124,567,890 parameters

[2/5] Creating dataset...
Created dummy dataset with 1000 samples
⚠️  IMPORTANT: Replace this with your actual image-text dataset!
✓ Dataset created with 1000 samples

[3/5] Creating optimizer...
✓ Total training steps: 625

[4/5] Training...
--------------------------------------------------------------------------------
Epoch 1 [0/62] Loss: 6.2145 Logit Scale: 14.27 LR: 0.000100
Epoch 1 [10/62] Loss: 5.8932 Logit Scale: 14.45 LR: 0.000099
...
✓ Epoch 1/10 completed | Avg Loss: 5.4521
...
✓ Checkpoint saved: ./checkpoints/checkpoint_epoch_5.pt
...
✓ Epoch 10/10 completed | Avg Loss: 2.1234
--------------------------------------------------------------------------------

[5/5] Saving final model...
✓ Checkpoint saved: ./checkpoints/checkpoint_epoch_10.pt

[Bonus] Testing inference...
✓ Similarity matrix (diagonal should be high):
  [0.8234, 0.7891, 0.8456, 0.7623]

========================================================================
✅ TRAINING COMPLETED SUCCESSFULLY!
Checkpoints saved to: ./checkpoints
========================================================================
```

---

## 📚 更多信息

- 完整文档: [REFACTORING_GUIDE.md](../REFACTORING_GUIDE.md)
- 详细说明: [README_DINOV3_QWEN.md](README_DINOV3_QWEN.md)
- 示例代码: [train_dinov3_qwen.py](train_dinov3_qwen.py)

---

## 💡 下一步

1. ✅ 运行测试训练验证环境
2. 📊 准备你的图像-文本数据集
3. 🔧 调整超参数优化性能
4. 🚀 扩展到分布式多GPU训练
5. 📈 添加验证和评估指标
