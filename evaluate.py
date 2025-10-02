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
    """
    return (data + 1) * (max_val - min_val) / 2 + min_val

def save_comparison_plot(input_img, target_img, pred_img, sample_idx, output_dir, stats):
    """
    为单个样本生成并保存包含输入、真值、预测和误差的四合一对比图。
    """
    # 反归一化
    target_min, target_max = stats['target']['min'], stats['target']['max']
    input_min, input_max = stats['input']['min'], stats['input']['max']
    
    input_img = denormalize(input_img, input_min, input_max)
    target_img = denormalize(target_img, target_min, target_max)
    pred_img = denormalize(pred_img, target_min, target_max)

    # 计算绝对误差
    abs_error = np.abs(pred_img - target_img)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'Sample #{sample_idx} Analysis', fontsize=16)

    # 1. 输入轨迹
    im1 = axes[0].imshow(input_img, cmap='gray')
    axes[0].set_title('Input Trajectory')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. 真实势场
    vmin, vmax = np.min(target_img), np.max(target_img)
    im2 = axes[1].imshow(target_img, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Ground Truth Potential')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. 预测势场
    im3 = axes[2].imshow(pred_img, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2].set_title('Predicted Potential')
    axes[2].axis('off')
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. 绝对误差
    im4 = axes[3].imshow(abs_error, cmap='inferno')
    axes[3].set_title('Absolute Error')
    axes[3].axis('off')
    fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f'sample_{sample_idx}_comparison.png')
    plt.savefig(save_path)
    plt.close(fig)
    logging.info(f"Saved comparison plot to '{save_path}'")


def evaluate():
    """
    主评估函数：加载模型，对指定数量的样本进行可视化和分析。
    """
    # 确保评估结果的输出目录存在
    os.makedirs(config.EVALUATION_DIR, exist_ok=True)
    
    # 设置设备 (CPU 或 GPU)
    device = torch.device(config.DEVICE)
    logging.info(f"Using device: {device}")

    # --- 1. 加载数据统计信息 ---
    try:
        with open(config.STATS_FILE, 'r') as f:
            stats = json.load(f)
    except FileNotFoundError:
        logging.error(f"Stats file not found at '{config.STATS_FILE}'. Please run train.py first to generate it.")
        return

    # --- 2. 准备数据集 ---
    transform = T.Compose([T.Resize((config.IMG_SIZE, config.IMG_SIZE), antialias=True)])
    val_dataset = ForceFieldDataset(
        input_dir=config.VALIDATION_B_DIR,
        target_dir=config.VALIDATION_A_DIR,
        stats=stats,
        transform=transform
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, # 一次只处理一个样本
        shuffle=False, 
        num_workers=0
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
    num_samples_to_save = config.NUM_VISUAL_SAMPLES if config.NUM_VISUAL_SAMPLES != -1 else len(val_loader)
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            if i >= num_samples_to_save:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            
            # 获取模型预测
            prediction_tensor = model(inputs)
            
            # --- 5. 转换为Numpy数组 ---
            input_normalized = inputs.squeeze(0).cpu().numpy()
            target_normalized = targets.squeeze(0).cpu().numpy()
            prediction_normalized = prediction_tensor.squeeze(0).cpu().numpy()
            
            # 确保数组是 2D 的
            if input_normalized.ndim == 3:
                input_normalized = input_normalized.squeeze(0)
            if target_normalized.ndim == 3:
                target_normalized = target_normalized.squeeze(0)
            if prediction_normalized.ndim == 3:
                prediction_normalized = prediction_normalized.squeeze(0)

            # --- 6. 生成并保存可视化结果 ---
            save_comparison_plot(
                input_img=input_normalized,
                target_img=target_normalized,
                pred_img=prediction_normalized,
                sample_idx=i,
                output_dir=config.EVALUATION_DIR,
                stats=stats
            )
            
    logging.info(f"Evaluation finished. Saved {num_samples_to_save} comparison images.")

if __name__ == '__main__':
    evaluate()