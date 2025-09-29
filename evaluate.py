# # evaluate.py

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
        # เราสามารถใช้ next() เพื่อดึงข้อมูลทีละชุด
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
 (Refactored Version)

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from scipy.stats import skew, kurtosis
import torchvision.transforms as T
import json

# Import all settings from the config file
import config
from dataset import ForceFieldDataset 
from model import UNet

# --- 1. Helper Functions ---

def calculate_statistics(image_np):
    """(CPU) Calculate RMS, Skewness, Kurtosis of an image."""
    rms = np.sqrt(np.mean(np.square(image_np)))
    sk = skew(image_np.flatten())
    kurt = kurtosis(image_np.flatten())
    return rms, sk, kurt

def calculate_psd_1d(image_np):
    """(CPU) Calculate the 1D radially averaged Power Spectral Density."""
    np_fft = np.fft.fft2(image_np)
    np_fft_shifted = np.fft.fftshift(np_fft)
    power_spectrum_2d = np.abs(np_fft_shifted)**2
    h, w = image_np.shape
    center = (h // 2, w // 2)
    y, x = np.indices((h, w))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    tbin = np.bincount(r.ravel(), power_spectrum_2d.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-9)
    return radial_profile[:int(min(h, w) / 2)]

def calculate_correlation_length_torch(image_tensor):
    """(GPU) High-performance calculation of autocorrelation and correlation length."""
    image_tensor = image_tensor - torch.mean(image_tensor)
    fft_val = torch.fft.fft2(image_tensor)
    psd = torch.abs(fft_val)**2
    autocorr = torch.fft.ifft2(psd).real
    autocorr = torch.fft.fftshift(autocorr)
    h, w = autocorr.shape
    profile = autocorr[h // 2, w // 2:]
    profile = profile / profile[0]
    profile_cpu = profile.cpu().numpy()
    try:
        corr_len_pixels = np.where(profile_cpu < 1/np.e)[0][0]
    except IndexError:
        corr_len_pixels = len(profile_cpu)
    return corr_len_pixels, profile_cpu

def denormalize(data, min_val, max_val):
    """Denormalize data from [0, 1] range to original scale."""
    return data * (max_val - min_val) + min_val

def evaluate():
    """Main evaluation function."""
    # --- 2. Setup from Config ---
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(config.EVALUATION_DIR, exist_ok=True)
    visual_output_dir = os.path.join(config.EVALUATION_DIR, "visual_comparisons")
    os.makedirs(visual_output_dir, exist_ok=True)

    # --- 3. Prepare Data and Model ---
    # Load normalization statistics
    try:
        with open(config.STATS_FILE, 'r') as f:
            stats = json.load(f)
        print(f"Successfully loaded stats from '{config.STATS_FILE}'")
    except FileNotFoundError:
        print(f"Error: Stats file not found at '{config.STATS_FILE}'. Please run train.py first.")
        return

    # Load model
    model = UNet(
        n_channels=config.MODEL_N_CHANNELS,
        n_classes=config.MODEL_N_CLASSES,
        bilinear=config.MODEL_BILINEAR
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
        model.eval()
        print(f"Successfully loaded model from '{config.BEST_MODEL_PATH}'")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{config.BEST_MODEL_PATH}'. Please complete training.")
        return

    # Prepare DataLoader
    data_transform = T.Compose([
        T.Resize((config.IMG_SIZE, config.IMG_SIZE), antialias=True),
    ])
    test_dataset = ForceFieldDataset(
        input_dir=config.TEST_B_DIR, 
        target_dir=config.TEST_A_DIR, 
        stats=stats, 
        transform=data_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- 4. Perform Evaluation ---
    metrics = {
        "mae": [], "real_rms": [], "pred_rms": [], "real_skew": [], "pred_skew": [],
        "real_kurt": [], "pred_kurt": [], "real_psd": [], "pred_psd": [],
        "real_corr_len": [], "pred_corr_len": [], "real_autocorr": [], "pred_autocorr": []
    }

    loop = tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating")
    for i, (inputs_normalized, targets_normalized) in loop:
        inputs_normalized = inputs_normalized.to(device)
        
        with torch.no_grad():
            predicted_normalized = model(inputs_normalized)

        # Denormalize to original physical scale for metrics calculation
        predicted_np = denormalize(predicted_normalized.squeeze().cpu().numpy(), stats['target']['min'], stats['target']['max'])
        target_np = denormalize(targets_normalized.squeeze().cpu().numpy(), stats['target']['min'], stats['target']['max'])
        
        # Calculate all metrics
        metrics["mae"].append(np.mean(np.abs(predicted_np - target_np)))
        
        real_rms, real_skew, real_kurt = calculate_statistics(target_np)
        pred_rms, pred_skew, pred_kurt = calculate_statistics(predicted_np)
        metrics["real_rms"].append(real_rms); metrics["pred_rms"].append(pred_rms)
        metrics["real_skew"].append(real_skew); metrics["pred_skew"].append(pred_skew)
        metrics["real_kurt"].append(real_kurt); metrics["pred_kurt"].append(pred_kurt)
        
        metrics["real_psd"].append(calculate_psd_1d(target_np))
        metrics["pred_psd"].append(calculate_psd_1d(predicted_np))

        real_len, real_ac = calculate_correlation_length_torch(torch.from_numpy(target_np).to(device))
        pred_len, pred_ac = calculate_correlation_length_torch(torch.from_numpy(predicted_np).to(device))
        metrics["real_corr_len"].append(real_len); metrics["pred_corr_len"].append(pred_len)
        metrics["real_autocorr"].append(real_ac); metrics["pred_autocorr"].append(pred_ac)

        # Save visualization samples
        if config.NUM_VISUAL_SAMPLES == -1 or i < config.NUM_VISUAL_SAMPLES:
            original_filename = os.path.basename(test_dataset.image_files[i])
            base_filename = os.path.splitext(original_filename)[0]
            input_img_denorm = denormalize(inputs_normalized.squeeze().cpu().numpy(), stats['input']['min'], stats['input']['max'])
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Comparison for {original_filename}', fontsize=16)
            axes[0].imshow(input_img_denorm, cmap='gray'); axes[0].set_title('Input Trajectory (Original Scale)'); axes[0].axis('off')
            axes[1].imshow(target_np, cmap='viridis'); axes[1].set_title('Ground Truth Force Field'); axes[1].axis('off')
            axes[2].imshow(predicted_np, cmap='viridis'); axes[2].set_title('Predicted Force Field'); axes[2].axis('off')
            save_path = os.path.join(visual_output_dir, f"comparison_{base_filename}.png")
            plt.savefig(save_path, bbox_inches='tight'); plt.close(fig)

    # --- 5. Output Evaluation Report ---
    print("\n\n" + "="*25 + " U-Net Model Quantitative Evaluation Report " + "="*25)
    print(f"\n[ Point-to-Point Error (on {len(test_loader)} samples) ]")
    print(f"  - Mean Absolute Error (MAE): {np.mean(metrics['mae']):.6f} (Std: {np.std(metrics['mae']):.6f})")
    print(f"\n[ Statistical Properties Comparison (Mean ± Std) ]")
    print(f"  - Root Mean Square (RMS):   True: {np.mean(metrics['real_rms']):.4f} ± {np.std(metrics['real_rms']):.4f} | Pred: {np.mean(metrics['pred_rms']):.4f} ± {np.std(metrics['pred_rms']):.4f}")
    print(f"  - Skewness:                 True: {np.mean(metrics['real_skew']):.4f} ± {np.std(metrics['real_skew']):.4f} | Pred: {np.mean(metrics['pred_skew']):.4f} ± {np.std(metrics['pred_skew']):.4f}")
    print(f"  - Kurtosis:                 True: {np.mean(metrics['real_kurt']):.4f} ± {np.std(metrics['real_kurt']):.4f} | Pred: {np.mean(metrics['pred_kurt']):.4f} ± {np.std(metrics['pred_kurt']):.4f}")
    print(f"\n[ Physical Properties Comparison (Mean ± Std) ]")
    print(f"  - Correlation Length (px):  True: {np.mean(metrics['real_corr_len']):.2f} ± {np.std(metrics['real_corr_len']):.2f} | Pred: {np.mean(metrics['pred_corr_len']):.2f} ± {np.std(metrics['pred_corr_len']):.2f}")
    print("="*80)

    # --- 6. Generate and Save Summary Plots ---
    plt.figure(figsize=(10, 6)); plt.hist(metrics["mae"], bins=20, edgecolor='black'); plt.title("Mean Absolute Error (MAE) Distribution"); plt.xlabel("MAE"); plt.ylabel("Sample Count"); plt.grid(True, linestyle='--', alpha=0.6); plt.savefig(os.path.join(config.EVALUATION_DIR, "mae_distribution.png")); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5)); stats_data = [("RMS", metrics["real_rms"], metrics["pred_rms"]),("Skewness", metrics["real_skew"], metrics["pred_skew"]),("Kurtosis", metrics["real_kurt"], metrics["pred_kurt"])];
    for ax, (title, real_data, pred_data) in zip(axes, stats_data):
        ax.boxplot([real_data, pred_data], labels=['True', 'Predicted']); ax.set_title(title + " Distribution Comparison"); ax.grid(True, linestyle='--', alpha=0.6)
    plt.suptitle("Statistical Properties Comparison", fontsize=16); plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(config.EVALUATION_DIR, "statistics_comparison.png")); plt.close()

    avg_real_psd = np.mean([p for p in metrics["real_psd"] if p is not None], axis=0)
    avg_pred_psd = np.mean([p for p in metrics["pred_psd"] if p is not None], axis=0)
    plt.figure(figsize=(10, 6)); plt.plot(avg_real_psd, label='Ground Truth', color='blue'); plt.plot(avg_pred_psd, label='Predicted', color='red', linestyle='--'); plt.title("Average 1D Power Spectral Density (PSD) Comparison"); plt.xlabel("Spatial Frequency"); plt.ylabel("Power"); plt.yscale('log'); plt.legend(); plt.grid(True, which="both", linestyle='--', alpha=0.6); plt.savefig(os.path.join(config.EVALUATION_DIR, "power_spectrum_comparison.png")); plt.close()

    min_len = min(len(p) for p in metrics["real_autocorr"] + metrics["pred_autocorr"])
    avg_real_ac = np.mean([p[:min_len] for p in metrics["real_autocorr"]], axis=0)
    avg_pred_ac = np.mean([p[:min_len] for p in metrics["pred_autocorr"]], axis=0)
    plt.figure(figsize=(10, 6)); plt.plot(avg_real_ac, label='Ground Truth', color='blue'); plt.plot(avg_pred_ac, label='Predicted', color='red', linestyle='--'); plt.axhline(1/np.e, color='green', linestyle=':', label='1/e Threshold'); plt.title("Average Autocorrelation Function"); plt.xlabel("Distance (pixels)"); plt.ylabel("Normalized Correlation"); plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.savefig(os.path.join(config.EVALUATION_DIR, "autocorrelation_comparison.png")); plt.close()

    print(f"\nQuantitative metric plots saved to '{config.EVALUATION_DIR}'.")
    print(f"Visual comparison images saved to '{visual_output_dir}'.")
    print("Evaluation finished.")

if __name__ == '__main__':
    evaluate()