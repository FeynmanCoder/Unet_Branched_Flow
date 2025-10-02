# 🔧 超算环境配置指南

## ❌ 问题: ModuleNotFoundError: No module named 'monai'

你的conda环境缺少必需的Python包。

---

## ✅ 解决方案

### 方法1: 使用安装脚本 (推荐)

```bash
# 提交安装任务
sbatch install_deps.sh

# 查看安装日志
cat slurm-*.out
```

### 方法2: 手动安装

```bash
# 1. 加载CUDA
module load cuda/12.6.0

# 2. 激活conda环境
source ~/software/miniconda3/etc/profile.d/conda.sh
conda activate ai_bf

# 3. 安装依赖包
pip install monai
pip install albumentations
pip install opencv-python-headless
pip install tqdm
pip install matplotlib

# 4. 验证安装
python -c "import monai; print('MONAI version:', monai.__version__)"
python -c "import albumentations; print('Albumentations OK')"
```

---

## 📦 必需的Python包

训练需要以下包:

| 包名 | 用途 |
|------|------|
| `monai` | MONAI UNet模型 |
| `albumentations` | 数据增强 |
| `opencv-python-headless` | 图像处理 |
| `tqdm` | 进度条 |
| `matplotlib` | 可视化 |
| `torch` | PyTorch (应该已安装) |
| `numpy` | 数值计算 (应该已安装) |

---

## 🔍 检查安装状态

```bash
# 激活环境
conda activate ai_bf

# 检查所有包
pip list | grep -E "monai|albumentations|opencv|tqdm|matplotlib|torch|numpy"
```

---

## 🚀 安装完成后

```bash
# 重新提交训练任务
sbatch train.sh
```

---

## ⚠️ 常见问题

### Q1: pip install 很慢?
```bash
# 使用清华镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple monai
```

### Q2: 权限错误?
```bash
# 安装到用户目录
pip install --user monai
```

### Q3: 版本冲突?
```bash
# 查看torch版本
python -c "import torch; print(torch.__version__)"

# MONAI需要torch>=1.9
# 如果torch版本太低,需要升级
```

---

## 📝 完整的环境配置命令

```bash
# 创建新的conda环境(如果需要)
conda create -n ai_bf python=3.10

# 激活环境
conda activate ai_bf

# 安装PyTorch (根据CUDA版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 安装其他依赖
pip install monai albumentations opencv-python-headless tqdm matplotlib numpy

# 验证
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import monai; print('MONAI:', monai.__version__)"
```

---

更新时间: 2025-10-01
