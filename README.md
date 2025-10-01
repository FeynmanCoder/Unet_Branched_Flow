# Unet_Branched_Flow

基于U-Net的粒子轨迹到势能分布的映射模型

## 🌿 分支说明

本项目有两个主要版本:

### 📌 main (推荐) - 分类方法
使用语义分割方法,将连续势能值离散化为8个类别
- ✅ 避免"纹理复制"问题
- ✅ 更稳定的训练
- ✅ 适合输入输出差异大的情况

### 📌 regression-version - 回归方法  
传统的回归方法,直接预测连续势能值
- 输出连续精确值
- 适合数据量大的情况
- 保留原始实现作为参考

**详细对比请查看**: [BRANCH_INFO.md](BRANCH_INFO.md)

---

## 🚀 快速开始 (main分支)

### 1. 环境检查
```bash
python check_environment.py
```

### 2. 训练模型
```bash
python train.py
```

### 3. 预测
```bash
python predict_classification.py
```

---

## 📚 文档

- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
- [TRAINING_IMPROVEMENTS.md](TRAINING_IMPROVEMENTS.md) - 技术改进说明
- [SOLUTION_SUMMARY.md](SOLUTION_SUMMARY.md) - 完整方案总结
- [BRANCH_INFO.md](BRANCH_INFO.md) - ⭐ 分支对比说明
- [CODE_REVIEW.md](CODE_REVIEW.md) - 代码审查报告

---

## 🔄 切换分支

### 切换到分类方法 (推荐)
```bash
git checkout main
```

### 切换到回归方法
```bash
git checkout regression-version
```

---

## 📊 主要改进 (main分支)

1. **回归→分类**: 离散化为8个类别,避免复制输入特征
2. **组合损失**: CrossEntropy + Dice Loss
3. **数据增强**: 翻转、旋转等
4. **更深网络**: MONAI UNet (6层)

---

## 📁 项目结构

```
.
├── train.py                    # 训练脚本
├── predict_classification.py   # 预测脚本(分类方法)
├── config.py                   # 配置文件
├── model.py                    # 模型定义
├── dataset.py                  # 数据集
├── evaluate.py                 # 评估脚本
├── check_environment.py        # 环境检查
└── docs/                       # 文档目录
    ├── QUICKSTART.md
    ├── TRAINING_IMPROVEMENTS.md
    ├── SOLUTION_SUMMARY.md
    └── BRANCH_INFO.md
```

---

## 🎯 核心特性

- ✅ 分类方法避免"纹理复制"
- ✅ Dice Loss改善空间连续性
- ✅ 数据增强提高泛化能力
- ✅ MONAI UNet更深的网络
- ✅ 完整的文档和检查工具

---

## 🔧 配置

主要配置在 `config.py`:
```python
NUM_CLASSES = 8           # 分类类别数
IMG_SIZE = 256           # 图像大小
BATCH_SIZE = 16          # 批次大小
LEARNING_RATE = 1e-4     # 学习率
EPOCHS = 200             # 训练轮数
```

---

## 📈 预期效果

| 指标 | 回归方法 | 分类方法(main) |
|------|----------|----------------|
| 轨迹残留 | ❌ 明显 | ✅ 很少 |
| 势能平滑 | ⚠️ 一般 | ✅ 好 |
| 训练稳定 | ⚠️ 一般 | ✅ 稳定 |
| 边界清晰 | ❌ 模糊 | ✅ 清晰 |

---

## 🐛 问题排查

### CUDA out of memory?
```python
# config.py
BATCH_SIZE = 8  # 减小批次
```

### 数据路径错误?
```python
# config.py
DATA_DIR = '你的数据路径'
```

### 环境依赖?
```bash
pip install monai albumentations torch torchvision tqdm
```

---

## 📞 支持

- Issues: https://github.com/FeynmanCoder/Unet_Branched_Flow/issues
- 查看文档: `QUICKSTART.md`, `BRANCH_INFO.md`

---

## 📝 更新日志

- **2025-10-01**: 添加分类方法 (main分支)
- **2025-10-01**: 保存回归方法到 regression-version 分支
- **2025-09**: 初始回归实现

---

**当前分支**: main (分类方法)  
**维护者**: FeynmanCoder
