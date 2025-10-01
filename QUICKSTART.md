# 🚀 快速开始指南

## 核心改进 - 解决"纹理复制"问题

主要改进策略:
1. ✅ **分类代替回归** - 避免模型复制轨迹特征
2. ✅ **Dice Loss** - 改善空间连续性
3. ✅ **数据增强** - 提高泛化能力  
4. ✅ **更深的网络** - MONAI UNet (6层)

### 为什么有效?
粒子轨迹和势能分布是完全不同的图像模式。直接回归会让模型倾向于"复制"输入特征。**分类方法**强制模型学习势能的空间结构,而不是像素值,从而避免了纹理复制问题。

## 一键运行

### 1. 训练模型
```bash
python train.py
```

**预期输出**:
```
准备数据集
开始训练
Epoch [1/200], Train Loss: 2.5432, Val Loss: 2.3456
Model saved at epoch 1 with val loss: 2.3456
...
```

### 2. 进行预测
```bash
python predict_classification.py
```

**输出**:
- `predictions/pred_*.npy` - 预测的势能图
- `predictions/comparisons/*_comparison.png` - 可视化对比图

## 关键配置

在 `config.py` 中:
```python
NUM_CLASSES = 8        # 势能离散化为8个等级
IMG_SIZE = 256        # 图像大小
BATCH_SIZE = 16       # 批次大小
LEARNING_RATE = 1e-4  # 学习率
EPOCHS = 200          # 训练轮数
```

## 数据目录结构

确保你的数据按以下结构组织:
```
data_2000/
├── trainA/          # 训练集 - 势能图(标签)
│   ├── 0001.npy
│   ├── 0002.npy
│   └── ...
├── trainB/          # 训练集 - 轨迹图(输入)
│   ├── 0001.npy
│   ├── 0002.npy
│   └── ...
├── validationA/     # 验证集 - 势能图
│   └── ...
├── validationB/     # 验证集 - 轨迹图
│   └── ...
├── testA/           # 测试集 - 势能图
│   └── ...
└── testB/           # 测试集 - 轨迹图
    └── ...
```

**注意**: A是势能(target), B是轨迹(input)

## 核心代码变化

### Dataset (train.py):
```python
# 关键改进: 离散化标签
mask = (mask * NUM_CLASSES)  # 转换为0-7的类别
mask[mask >= NUM_CLASSES] = NUM_CLASSES - 1
mask[mask < 0] = 0
```

### Loss Function (train.py):
```python
def combined_loss(pred, target):
    ce = CrossEntropyLoss(pred, target)  # 分类
    dice = dice_loss_multiclass(pred, target)  # 边界
    return ce + dice
```

### Model (train.py):
```python
model = UNet(
    in_channels=1,
    out_channels=NUM_CLASSES,  # 输出8个类别!
    spatial_dims=2,
    channels=(32, 64, 128, 256, 320, 320),  # 6层深度
    strides=(2, 2, 2, 2, 2),
)
```

### Prediction (predict_classification.py):
```python
outputs = model(images)  # (B, 8, H, W)
pred_classes = outputs.argmax(dim=1)  # 获取类别
pred_values = pred_classes.float() / NUM_CLASSES  # 转为[0,1]
```

## 监控训练

### 正常训练应该看到:
```
Epoch [1/200], Train Loss: 2.5432, Val Loss: 2.3456
Epoch [2/200], Train Loss: 2.1234, Val Loss: 2.0123
Epoch [3/200], Train Loss: 1.8765, Val Loss: 1.7654
...
Epoch [50/200], Train Loss: 0.8234, Val Loss: 0.7654
Model saved at epoch 50 with val loss: 0.7654
```

### 异常情况:
- ❌ Loss不下降 → 检查学习率,降低到5e-5
- ❌ Loss=NaN → 检查数据范围,是否有异常值
- ❌ Val Loss上升 → 正常,早停机制会保存最佳模型

## 评估结果

训练完成后,用以下方式检查:

### 1. 查看对比图
```bash
# 在 predictions/comparisons/ 目录下
# 打开任意 *_comparison.png 文件
```

### 2. 定量评估 (如果需要)
```python
# 计算MSE
from sklearn.metrics import mean_squared_error

pred = np.load('predictions/pred_0001.npy')
true = np.load('data_2000/testA/0001.npy')
mse = mean_squared_error(true.flatten(), pred.flatten())
print(f'MSE: {mse}')
```

## 故障排除

### 问题1: ImportError: No module named 'monai'
```bash
pip install monai
```

### 问题2: CUDA out of memory
降低batch size:
```python
# config.py
BATCH_SIZE = 8  # 或更小
```

### 问题3: 预测结果仍有轨迹残留
尝试:
1. 增加类别数: `NUM_CLASSES = 16`
2. 增加Dice权重: `return ce + 2*dice`
3. 训练更多轮次

### 问题4: 找不到数据文件
检查 `config.py` 中的路径是否正确:
```python
DATA_DIR = '/lustre/home/2400011491/data/ai_train_data/data_2000'
```

## 进阶调优

### 如果效果不错,可以尝试:
1. **更多数据增强**:
   ```python
   A.RandomBrightnessContrast(p=0.2)
   A.GaussianBlur(blur_limit=3, p=0.2)
   ```

2. **调整类别数**:
   ```python
   NUM_CLASSES = 16  # 更细粒度
   ```

3. **使用余弦学习率衰减**:
   ```python
   scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
   ```

## 📖 更多文档

- 详细技术说明: `TRAINING_IMPROVEMENTS.md`
- 完整方案说明: `SOLUTION_SUMMARY.md`

---

**就这么简单!运行 `python train.py` 开始训练吧!** 🎉
