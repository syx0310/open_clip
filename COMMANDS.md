# 完整命令列表：从安装到执行

## 📦 方法1: 一键运行（推荐）

```bash
cd /home/user/open_clip
bash examples/setup_and_run.sh
```

按照交互式提示选择即可！

---

## 🔧 方法2: 手动安装和运行

### 步骤1: 安装依赖

```bash
# 进入项目目录
cd /home/user/open_clip

# 安装PyTorch (根据你的CUDA版本选择)
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安装Transformers和其他依赖
pip install transformers accelerate pillow

# 安装OpenCLIP (开发模式)
pip install -e .
```

### 步骤2: 验证安装

```bash
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"
python -c "import torch; print(f'✓ CUDA: {torch.cuda.is_available()}')"
python -c "from open_clip.base_model import TransformersVisionEncoder; print('✓ OpenCLIP OK')"
```

### 步骤3: 运行训练

#### 🎯 1:1 标准训练（DinoV2 + Qwen）

```bash
python examples/train_dinov3_qwen.py \
    --mode 1to1 \
    --vision-model facebook/dinov2-base \
    --text-model Qwen/Qwen2-0.5B \
    --embed-dim 768 \
    --batch-size 16 \
    --epochs 10 \
    --learning-rate 1e-4 \
    --num-samples 1000 \
    --device cuda
```

#### 🔄 1:N 训练（1个文本 + 多个视觉编码器）

```bash
python examples/train_dinov3_qwen.py \
    --mode 1text_nvision \
    --num-vision 3 \
    --vision-model facebook/dinov2-base \
    --text-model Qwen/Qwen2-0.5B \
    --embed-dim 768 \
    --batch-size 8 \
    --epochs 10 \
    --aggregation mean \
    --device cuda
```

#### 🔁 N:1 训练（多个文本编码器 + 1个视觉）

```bash
python examples/train_dinov3_qwen.py \
    --mode ntext_1vision \
    --num-text 2 \
    --vision-model facebook/dinov2-base \
    --text-model Qwen/Qwen2-0.5B \
    --embed-dim 768 \
    --batch-size 8 \
    --epochs 10 \
    --aggregation mean \
    --device cuda
```

---

## 🚀 使用大模型训练

### DinoV2-Giant + Qwen2-7B (需要40GB+ GPU)

```bash
python examples/train_dinov3_qwen.py \
    --mode 1to1 \
    --vision-model facebook/dinov2-giant \
    --text-model Qwen/Qwen2-7B \
    --embed-dim 1536 \
    --batch-size 4 \
    --epochs 10 \
    --learning-rate 5e-5 \
    --device cuda
```

---

## 🧪 快速测试（小数据集）

```bash
python examples/train_dinov3_qwen.py \
    --mode 1to1 \
    --vision-model facebook/dinov2-small \
    --text-model Qwen/Qwen2-0.5B \
    --batch-size 32 \
    --epochs 3 \
    --num-samples 100 \
    --log-interval 2
```

---

## 📊 查看训练结果

```bash
# 查看检查点
ls -lh checkpoints/

# 加载和测试检查点
python -c "
import torch
checkpoint = torch.load('checkpoints/checkpoint_epoch_10.pt')
print(f'Epoch: {checkpoint[\"epoch\"]}')
print(f'Loss: {checkpoint[\"loss\"]:.4f}')
"
```

---

## 🔍 所有可用参数

```bash
python examples/train_dinov3_qwen.py --help
```

输出：
```
usage: train_dinov3_qwen.py [-h]
    [--mode {1to1,1text_nvision,ntext_1vision}]
    [--num-vision NUM_VISION]
    [--num-text NUM_TEXT]
    [--vision-model VISION_MODEL]
    [--text-model TEXT_MODEL]
    [--embed-dim EMBED_DIM]
    [--vision-pooler {cls,mean,max}]
    [--text-pooler {cls,mean,max}]
    [--batch-size BATCH_SIZE]
    [--epochs EPOCHS]
    [--learning-rate LEARNING_RATE]
    [--weight-decay WEIGHT_DECAY]
    [--max-grad-norm MAX_GRAD_NORM]
    [--num-samples NUM_SAMPLES]
    [--image-size IMAGE_SIZE]
    [--max-text-length MAX_TEXT_LENGTH]
    [--num-workers NUM_WORKERS]
    [--device DEVICE]
    [--checkpoint-dir CHECKPOINT_DIR]
    [--log-interval LOG_INTERVAL]
    [--save-interval SAVE_INTERVAL]
    [--aggregation {mean,max,weighted}]
```

---

## 🎨 可用模型列表

### Vision Models (HuggingFace)
```bash
--vision-model facebook/dinov2-small       # 22M params
--vision-model facebook/dinov2-base        # 86M params (推荐)
--vision-model facebook/dinov2-large       # 304M params
--vision-model facebook/dinov2-giant       # 1.1B params
--vision-model google/vit-base-patch16-224 # ViT-B/16
--vision-model microsoft/swin-base-patch4-window7-224
```

### Text Models (HuggingFace)
```bash
--text-model Qwen/Qwen2-0.5B              # 0.5B params (推荐测试)
--text-model Qwen/Qwen2-1.5B              # 1.5B params
--text-model Qwen/Qwen2-7B                # 7B params (推荐生产)
--text-model bert-base-uncased             # BERT-Base
--text-model roberta-base                  # RoBERTa-Base
```

---

## 🐛 常见问题解决

### 1. 内存不足 (OOM)

```bash
# 减小批量大小
python examples/train_dinov3_qwen.py --batch-size 4

# 使用更小的模型
python examples/train_dinov3_qwen.py \
    --vision-model facebook/dinov2-small \
    --text-model Qwen/Qwen2-0.5B
```

### 2. CUDA不可用

```bash
# 使用CPU
python examples/train_dinov3_qwen.py --device cpu --batch-size 4
```

### 3. 模型下载慢

```bash
# 使用HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com
python examples/train_dinov3_qwen.py ...
```

### 4. 检查GPU使用情况

```bash
# 监控GPU
watch -n 1 nvidia-smi

# 在另一个终端运行训练
python examples/train_dinov3_qwen.py ...
```

---

## 📝 下一步

1. **替换为真实数据**：编辑 `train_dinov3_qwen.py` 中的 `DummyImageTextDataset`
2. **调整超参数**：根据你的数据集大小和GPU内存调整
3. **分布式训练**：使用torchrun进行多GPU训练
4. **评估指标**：添加验证集和评估指标

---

## 📚 完整文档

- 快速开始: [examples/QUICKSTART.md](examples/QUICKSTART.md)
- 详细指南: [examples/README_DINOV3_QWEN.md](examples/README_DINOV3_QWEN.md)
- 重构指南: [REFACTORING_GUIDE.md](REFACTORING_GUIDE.md)

---

## ✅ 验证一切正常

运行完整测试套件：

```bash
python examples/test_dinov3_qwen.py
```

这会测试所有三种模式（1:1, 1:N, N:1）并验证所有功能。

---

**就是这样！享受训练吧！🎉**
