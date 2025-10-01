# 训练改进说明

## 问题描述
原始模型在从粒子轨迹图反推势能分布图时,生成的图像总是带有粒子轨迹的特征,而不是真正学到了从轨迹到势能的映射关系。

## 核心改进

### 1. 将回归问题转换为分类问题
**原理**: 粒子轨迹和势能分布是两种完全不同的图像模式。直接使用回归(输出连续值)会让模型倾向于"复制"输入图像的特征。

**改进**: 将连续的势能值离散化为8个类别,转换为语义分割任务。

```python
# 在 Dataset 的 __getitem__ 方法中
mask = (mask * NUM_CLASSES)  # 离散化为0-7的8个类别
mask[mask >= NUM_CLASSES] = NUM_CLASSES - 1
mask[mask < 0] = 0
```

**为什么有效**:
- 分类任务迫使模型学习势能的"等级"而不是精确值
- 模型无法简单地复制输入图像的像素值
- 语义分割更关注空间结构和边界,与势能场的物理特性匹配

### 2. 组合损失函数: CrossEntropy + Dice Loss

**交叉熵损失** (`CrossEntropyLoss`):
- 确保模型正确分类每个像素的势能等级
- 对所有类别一视同仁

**Dice Loss**:
- 源自医学图像分割,专门处理空间连续性
- 关注整体形状和边界的准确性
- 处理类别不平衡问题

```python
def combined_loss(pred, target):
    ce = ce_loss(pred, target)
    dice = dice_loss_multiclass(pred, target)
    return ce + dice
```

**为什么有效**:
- 两种损失函数从不同角度优化模型
- Dice Loss强制模型关注势能场的整体空间分布
- 组合使用避免模型"偷懒"复制输入纹理

### 3. 数据增强

```python
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),      # 水平翻转
    A.VerticalFlip(p=0.5),        # 垂直翻转  
    A.RandomRotate90(p=0.5),      # 随机90度旋转
    ToTensorV2(),
])
```

**为什么有效**:
- 增加训练数据的多样性
- 模型看到更多不同方向的轨迹-势能对应关系
- 提高泛化能力,减少对特定方向的过拟合

### 4. 使用更深的MONAI UNet

```python
model = UNet(
    in_channels=1,
    out_channels=NUM_CLASSES,
    spatial_dims=2,
    channels=(32, 64, 128, 256, 320, 320),  # 6层深度
    strides=(2, 2, 2, 2, 2),
)
```

**为什么有效**:
- 比标准4层UNet更深,有6层
- 更强的特征提取能力
- 能够学习更复杂的轨迹-势能映射关系
- 更大的感受野可以捕捉全局结构

## 使用方法

1. 确保 `config.py` 中已设置 `NUM_CLASSES = 8`
2. 运行训练:
```bash
python train.py
```

3. 模型输出是8个类别的概率分布,需要转换回连续值:
```python
# 在推理时
output = model(input)
pred_class = output.argmax(dim=1)  # 获取预测类别 (0-7)
pred_value = pred_class.float() / NUM_CLASSES  # 转换为 [0, 1] 范围
```

## 关键配置参数

- `NUM_CLASSES = 8`: 势能离散化的类别数
- `IMG_SIZE = 256`: 训练图像大小
- `BATCH_SIZE = 16`: 批次大小
- `LEARNING_RATE = 1e-4`: 学习率
- `EPOCHS = 200`: 训练轮数

## 预期效果

使用这些改进后,模型应该:
1. ✅ 生成的势能图不再包含明显的粒子轨迹纹理
2. ✅ 学会了势能场的平滑空间分布
3. ✅ 边界和梯度更加准确
4. ✅ 验证损失稳定下降

## 参考

这些改进基于医学图像分割和语义分割领域的成熟技术:
- Dice Loss: "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
- MONAI UNet: 专为医学图像设计的优化UNet实现
- 分类而非回归: 在不确定性较大的逆问题中,分类通常比回归更稳定
