# config.py

"""
这是一个集中的配置文件，用于管理模型训练的所有超参数和路径。
修改此文件中的值即可调整训练行为，无需更改源代码。
"""

# -------------------
# 数据与路径配置 (Data and Path Configuration)
# -------------------
DATA_CONFIG = {
    # 数据集根目录。请根据你的实际环境修改此路径。
    "data_path": "/lustre/home/2400011491/data/ai_train_data/data_20000",
    
    # 归一化统计文件的保存路径。
    "stats_file": "data_stats.json",
    
    # 最佳模型的保存路径。
    "save_path": "checkpoints/unet_wider_model.pth",
    
    # 训练和验证时，图像被统一调整到的尺寸。
    "img_size": 256,
}

# -------------------
# 模型配置 (Model Configuration)
# -------------------
MODEL_CONFIG = {
    # U-Net模型的基础通道数。增加此值可以“加宽”网络，提升模型容量。
    # 64 是原始大小, 96 或 128 是更宽的选择。
    "base_channels": 96,
    
    # 输入图像的通道数 (例如，灰度图为1，RGB图为3)。
    "n_channels": 1,
    
    # 模型输出的通道数。
    "n_classes": 1,
    
    # 是否在解码器（上采样部分）使用双线性插值。
    # False 表示使用转置卷积 (ConvTranspose2d)。
    "bilinear": False,
}

# -------------------
# 训练超参数配置 (Training Hyperparameters)
# -------------------
TRAIN_CONFIG = {
    # 最大训练周期数。
    "epochs": 200,
    
    # 批次大小。如果你的GPU显存充足，可以适当增加此值，例如 8, 16, 32。
    "batch_size": 8,
    
    # 优化器的学习率。
    "lr": 1e-4,
    
    # 优化器的权重衰减 (L2正则化)。
    "weight_decay": 1e-5,
    
    # 早停机制的“耐心值”。
    # 如果验证集损失连续 N 个周期没有改善，则停止训练。
    # 对于更大的模型，可以适当增加此值，例如 15 或 20。
    "patience": 15,
    
    # 混合损失中，像素损失 (L1 Loss) 的权重。
    "lambda_pixel": 1.0,
    
    # 混合损失中，物理损失 (谱损失) 的权重。
    "lambda_physics": 0.1,
}
