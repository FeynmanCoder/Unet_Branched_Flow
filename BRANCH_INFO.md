# 🔀 分支说明文档

## 📋 分支概览

本仓库现在有两个主要版本的实现:

### 1. 🌿 `main` 分支 (当前主分支) - **分类方法**

**提交**: `ee14d0e` - "改用分类而非回归来预测势能分布"

**方法**: 将连续势能值离散化为8个类别,使用语义分割方法

**核心特点**:
- ✅ 输出 8 个类别 (`NUM_CLASSES = 8`)
- ✅ 损失函数: CrossEntropy + Dice Loss
- ✅ 数据增强: HorizontalFlip, VerticalFlip, RandomRotate90
- ✅ MONAI UNet (6层深度)
- ✅ 标签离散化: `mask = mask * NUM_CLASSES`

**优势**:
- 避免模型直接复制输入轨迹特征
- 更关注势能场的空间结构
- 更稳定的训练过程
- 更好的泛化能力

**适用场景**: 
- 输入输出差异大的情况
- 需要关注空间结构而非精确数值

---

### 2. 🌿 `regression-version` 分支 - **回归方法**

**提交**: `531b2ed` - "参考旧代码对源代码做了一些优化"

**方法**: 直接回归连续势能值

**核心特点**:
- 输出 1 个通道 (连续值)
- 损失函数: MSE / L1 Loss
- 标准 UNet
- 直接预测连续势能值

**优势**:
- 输出是连续的精确值
- 更简单直接
- 适合数据充足的情况

**劣势**:
- 可能出现"纹理复制"问题
- 容易过拟合到输入特征

**适用场景**:
- 数据量非常大(>10000对)
- 输入输出相似度较高
- 需要精确数值

---

## 🔄 如何切换分支

### 切换到分类方法 (main分支)
```bash
git checkout main
```

### 切换到回归方法
```bash
git checkout regression-version
```

### 查看当前所在分支
```bash
git branch
```

### 查看所有分支
```bash
git branch -a
```

---

## 📊 两个版本的对比

| 特性 | main (分类) | regression-version (回归) |
|------|-------------|---------------------------|
| 输出类型 | 8个类别 | 1个连续值 |
| 损失函数 | CrossEntropy + Dice | MSE/L1 |
| 网络深度 | 6层 (MONAI UNet) | 4层 (标准UNet) |
| 数据增强 | ✅ 丰富 | ❌ 较少 |
| 训练稳定性 | ✅ 高 | ⚠️ 中等 |
| 避免纹理复制 | ✅ 是 | ❌ 否 |
| 输出精度 | 离散(8级) | 连续 |
| 适用数据量 | 中等(1000+) | 大量(10000+) |

---

## 📁 主要文件差异

### main 分支新增文件:
- `QUICKSTART.md` - 快速开始指南
- `TRAINING_IMPROVEMENTS.md` - 改进技术说明
- `SOLUTION_SUMMARY.md` - 方案总结
- `CODE_REVIEW.md` - 代码审查报告
- `FINAL_CHECK.md` - 最终检查报告
- `check_environment.py` - 环境检查脚本
- `predict_classification.py` - 分类方法预测脚本

### train.py 主要差异:

**regression-version**:
```python
# 直接使用连续mask
mask = np.load(mask_path)

# MSE损失
loss = F.mse_loss(outputs, masks)

# 标准UNet
model = UNet(n_channels=1, n_classes=1)
```

**main** (分类):
```python
# 离散化mask
mask = (mask * NUM_CLASSES)
mask[mask >= NUM_CLASSES] = NUM_CLASSES - 1

# 组合损失
loss = CrossEntropyLoss + DiceLoss

# MONAI UNet
model = UNet(
    in_channels=1,
    out_channels=NUM_CLASSES,
    channels=(32, 64, 128, 256, 320, 320)
)
```

---

## 🎯 推荐使用

### 推荐使用 `main` 分支 (分类方法) 如果:
- ✅ 粒子轨迹和势能分布差异很大
- ✅ 遇到"纹理复制"问题
- ✅ 训练数据量中等 (1000-5000对)
- ✅ 更关注空间结构而非精确数值

### 推荐使用 `regression-version` 分支 (回归方法) 如果:
- ✅ 有大量训练数据 (>10000对)
- ✅ 需要连续的精确数值输出
- ✅ 输入输出相似度较高
- ✅ 有足够的正则化和数据增强

---

## 🔧 实验建议

建议先在 `main` 分支尝试分类方法:
1. 训练50个epoch观察效果
2. 如果效果好,继续训练到收敛
3. 如果需要更高精度,可以尝试:
   - 增加类别数: `NUM_CLASSES = 16`
   - 或切换到 `regression-version` 对比效果

---

## 📝 版本历史

| 日期 | 分支 | 说明 |
|------|------|------|
| 2025-10-01 | regression-version | 保存原始回归方法 |
| 2025-10-01 | main | 改进为分类方法 |

---

## 🆘 需要帮助?

- 查看 `QUICKSTART.md` (main分支) - 快速开始
- 查看 `TRAINING_IMPROVEMENTS.md` (main分支) - 技术细节
- 查看原始训练脚本 (regression-version分支)

---

## 🔗 远程仓库

两个分支都已推送到GitHub:
- https://github.com/FeynmanCoder/Unet_Branched_Flow (main分支)
- https://github.com/FeynmanCoder/Unet_Branched_Flow/tree/regression-version

---

**更新时间**: 2025年10月1日  
**维护者**: FeynmanCoder
