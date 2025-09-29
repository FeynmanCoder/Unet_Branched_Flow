# evaluate.py

import os
import json
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as T

# 导入项目配置和自定义模块
import config
from dataset import ForceFieldDataset
from model import UNet

# 设置日志记录，方便观察脚本运行过程
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def denormalize(data, min_val, max_val):
    """
    将数据从 [-1, 1] 的范围反归一化到原始范围。
    这是 dataset.py 中 normalize 函数的逆操作。
    公式: original = (normalized + 1) * (max - min) / 2 + min
    """
    return (data + 1) * (max_val - min_val) / 2 + min_val

def plot_results(target, prediction, sample_idx, output_dir):
    """
    为单个样本生成并保存详细的对比图和误差分布图。
    
    Args:
        target (np.array): 真实的目标物理场数据 (已反归一化)。
        prediction (np.array): 模型预测的物理场数据 (已反归一化)。
        sample_idx (int): 当前样本的索引，用于命名文件。
        output_dir (string): 保存图像的目录。
    """
    # --- 1. 计算误差 ---
    # 绝对误差
    absolute_error = np.abs(prediction - target)
    
    # 百分比误差
    # 为避免分母为零，在分母上加一个很小的数 (epsilon)
    epsilon = 1e-8
    percentage_error = absolute_error / (np.abs(target) + epsilon) * 100
    
    # --- 2. 绘制四合一对比图 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Sample #{sample_idx} - Prediction vs. Target Analysis', fontsize=16)

    # 绘图用的色彩范围，基于真实值的范围
    vmin, vmax = np.min(target), np.max(target)

    # a) 真实场 (Ground Truth)
    im1 = axes[0, 0].imshow(target, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('Ground Truth (Target)')
    axes[0, 0].axis('off')
    fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # b) 预测场 (Prediction)
    im2 = axes[0, 1].imshow(prediction, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Model Prediction')
    axes[0, 1].axis('off')
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # c) 绝对误差
    im3 = axes[1, 0].imshow(absolute_error, cmap='inferno')
    axes[1, 0].set_title('Absolute Error: |Prediction - Target|')
    axes[1, 0].axis('off')
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # d) 百分比误差 (为了可视化效果，将上限设为100%)
    im4 = axes[1, 1].imshow(percentage_error, cmap='inferno', vmin=0, vmax=100)
    axes[1, 1].set_title('Percentage Error (%) [Capped at 100%]')
    axes[1, 1].axis('off')
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    comparison_path = os.path.join(output_dir, f'sample_{sample_idx}_comparison.png')
    plt.savefig(comparison_path)
    plt.close(fig)
    logging.info(f"Saved comparison plot to '{comparison_path}'")

    # --- 3. 绘制百分比误差的分布直方图 ---
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    # 移除极端值以便更好地观察分布主体
    clipped_errors = percentage_error[percentage_error < 200] # 只看200%以下的误差
    ax_hist.hist(clipped_errors.flatten(), bins=100, range=(0, 100)) # 绘制0-100%范围的直方图
    ax_hist.set_title(f'Sample #{sample_idx} - Percentage Error Distribution')
    ax_hist.set_xlabel('Percentage Error (%)')
    ax_hist.set_ylabel('Pixel Count')
    ax_hist.grid(True, alpha=0.5)
    
    histogram_path = os.path.join(output_dir, f'sample_{sample_idx}_error_distribution.png')
    plt.savefig(histogram_path)
    plt.close(fig_hist)
    logging.info(f"Saved error distribution plot to '{histogram_path}'")


def evaluate():
    """
    主评估函数：加载模型，对指定数量的样本进行可视化和分析。
    """
    # 确保评估结果的输出目录存在
    os.makedirs(config.EVALUATION_OUTPUT_DIR, exist_ok=True)
    
    # 设置设备 (CPU 或 GPU)
    device = torch.device(config.DEVICE)
    logging.info(f"Using device: {device}")

    # --- 1. 加载数据统计信息 ---
    # 这是反归一化所必需的
    try:
        with open(config.STATS_FILE, 'r') as f:
            stats = json.load(f)
        target_min = stats['target']['min']
        target_max = stats['target']['max']
    except FileNotFoundError:
        logging.error(f"Stats file not found at '{config.STATS_FILE}'. Please run train.py first to generate it.")
        return

    # --- 2. 准备数据集 ---
    # 使用验证集进行评估
    transform = T.Compose([T.Resize((config.IMG_SIZE, config.IMG_SIZE), antialias=True)])
    val_dataset = ForceFieldDataset(
        input_dir=config.VALIDATION_B_DIR,
        target_dir=config.VALIDATION_A_DIR,
        stats=stats,
        transform=transform
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, # 一次只处理一个样本，方便逐个分析
        shuffle=False, 
        num_workers=0 # 在Windows上设为0通常更稳定
    )

    # --- 3. 加载模型 ---
    model = UNet(
        n_channels=config.MODEL_N_CHANNELS,
        n_classes=config.MODEL_N_CLASSES,
        bilinear=config.MODEL_BILINEAR
    ).to(device)

    try:
        model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
        logging.info(f"Model loaded from '{config.BEST_MODEL_PATH}'")
    except FileNotFoundError:
        logging.error(f"Model file not found at '{config.BEST_MODEL_PATH}'. Please ensure the model is trained and saved.")
        return
        
    model.eval() # 设置为评估模式

    # --- 4. 循环处理样本并绘图 ---
    with torch.no_grad():
        # iter(val_loader) 将数据加载器变成一个迭代器
        # 我们可以使用 next() 来逐个获取数据
        data_iter = iter(val_loader)
        for i in range(config.NUM_EVAL_IMAGES):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                logging.warning(f"Ran out of samples in validation set. Only processed {i} images.")
                break

            inputs = inputs.to(device)
            
            # 获取模型预测
            prediction_tensor = model(inputs)
            
            # --- 5. 反归一化并转换为Numpy数组 ---
            # 将Tensor从GPU移到CPU，并移除batch维度和channel维度
            prediction_normalized = prediction_tensor.squeeze().cpu().numpy()
            target_normalized = targets.squeeze().cpu().numpy()
            
            # 执行反归一化
            prediction_denormalized = denormalize(prediction_normalized, target_min, target_max)
            target_denormalized = denormalize(target_normalized, target_min, target_max)

            # --- 6. 生成并保存可视化结果 ---
            plot_results(
                target=target_denormalized,
                prediction=prediction_denormalized,
                sample_idx=i,
                output_dir=config.EVALUATION_OUTPUT_DIR
            )
            
    logging.info("Evaluation finished.")

if __name__ == '__main__':
    evaluate()