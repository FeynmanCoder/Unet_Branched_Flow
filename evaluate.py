# evaluate.py (最終完整版)

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

# 確保 dataset.py 和 model.py 是最新的版本
from dataset import ForceFieldDataset 
from model import UNet

# --- 1. 輔助函數 ---

def calculate_statistics(image_np):
    """(CPU) 計算圖像的 RMS, Skewness, Kurtosis"""
    rms = np.sqrt(np.mean(np.square(image_np)))
    sk = skew(image_np.flatten())
    kurt = kurtosis(image_np.flatten())
    return rms, sk, kurt

def calculate_psd_1d(image_np):
    """(CPU) 計算圖像的1D徑向平均功率譜密度"""
    np_fft = np.fft.fft2(image_np)
    np_fft_shifted = np.fft.fftshift(np_fft)
    power_spectrum_2d = np.abs(np_fft_shifted)**2
    h, w = image_np.shape
    center = (h // 2, w // 2)
    y, x = np.indices((h, w))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), power_spectrum_2d.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-9)
    return radial_profile[:int(min(h, w) / 2)]

def calculate_correlation_length_torch(image_tensor):
    """(GPU) 高性能計算自相關和關聯長度"""
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
    """將 [0, 1] 範圍的數據反歸一化到原始尺度"""
    return data * (max_val - min_val) + min_val

# --- 2. 參數設定 ---
PROJECT_DIR = '.'
DATA_DIR = '/lustre/home/2400011491/data/ai_train_data/data_20000'
TEST_A_DIR = os.path.join(DATA_DIR, 'testA') 
TEST_B_DIR = os.path.join(DATA_DIR, 'testB') 
STATS_FILE = 'data_stats.json'

MODEL_PATH = os.path.join(PROJECT_DIR, "unet_model.pth") # 確保加載最終模型
OUTPUT_DIR = os.path.join(PROJECT_DIR, "evaluation_results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_SIZE = 256
NUM_VISUAL_SAMPLES = 20  # 設定要儲存的可視化樣本數量 (-1 代表所有)

# --- 3. 準備數據與模型 ---
print(f"使用設備: {DEVICE}")

# 創建輸出文件夾
os.makedirs(OUTPUT_DIR, exist_ok=True)
VISUAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "visual_comparisons")
os.makedirs(VISUAL_OUTPUT_DIR, exist_ok=True)

# 加載歸一化統計數據
try:
    with open(STATS_FILE, 'r') as f:
        stats = json.load(f)
    print(f"成功加載統計文件 '{STATS_FILE}'")
except FileNotFoundError:
    print(f"錯誤: 未找到 '{STATS_FILE}'。請先運行 'train.py' 來自動生成此文件。")
    exit()

# 加載模型
model = UNet(n_channels=1, n_classes=1).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    model.eval()
    print(f"成功加載模型: '{MODEL_PATH}'")
except FileNotFoundError:
    print(f"錯誤: 未找到模型文件 '{MODEL_PATH}'。請先完成訓練。")
    exit()

# 準備數據加載器
data_transform = T.Compose([
    T.Resize((TARGET_SIZE, TARGET_SIZE), antialias=True),
])
test_dataset = ForceFieldDataset(dir_A=TEST_A_DIR, dir_B=TEST_B_DIR, stats=stats, transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# --- 4. 執行評估 ---
metrics = {
    "mae": [], "real_rms": [], "pred_rms": [], "real_skew": [], "pred_skew": [],
    "real_kurt": [], "pred_kurt": [], "real_psd": [], "pred_psd": [],
    "real_corr_len": [], "pred_corr_len": [], "real_autocorr": [], "pred_autocorr": []
}

loop = tqdm(enumerate(test_loader), total=len(test_loader), desc="正在評估")
for i, (inputs_normalized, targets_normalized) in loop:
    inputs_normalized = inputs_normalized.to(DEVICE)
    
    with torch.no_grad():
        predicted_normalized = model(inputs_normalized)

    # --- 反歸一化 ---
    predicted_np = denormalize(predicted_normalized.squeeze().cpu().numpy(), stats['target']['min'], stats['target']['max'])
    target_np = denormalize(targets_normalized.squeeze().cpu().numpy(), stats['target']['min'], stats['target']['max'])
    
    # --- 指標計算 (在真實物理尺度上進行) ---
    metrics["mae"].append(np.mean(np.abs(predicted_np - target_np)))
    
    real_rms, real_skew, real_kurt = calculate_statistics(target_np)
    pred_rms, pred_skew, pred_kurt = calculate_statistics(predicted_np)
    metrics["real_rms"].append(real_rms); metrics["pred_rms"].append(pred_rms)
    metrics["real_skew"].append(real_skew); metrics["pred_skew"].append(pred_skew)
    metrics["real_kurt"].append(real_kurt); metrics["pred_kurt"].append(pred_kurt)
    
    metrics["real_psd"].append(calculate_psd_1d(target_np))
    metrics["pred_psd"].append(calculate_psd_1d(predicted_np))

    real_len, real_ac = calculate_correlation_length_torch(torch.from_numpy(target_np).to(DEVICE))
    pred_len, pred_ac = calculate_correlation_length_torch(torch.from_numpy(predicted_np).to(DEVICE))
    metrics["real_corr_len"].append(real_len); metrics["pred_corr_len"].append(pred_len)
    metrics["real_autocorr"].append(real_ac); metrics["pred_autocorr"].append(pred_ac)

    # --- 可視化輸出 ---
    if NUM_VISUAL_SAMPLES == -1 or i < NUM_VISUAL_SAMPLES:
        original_filename = test_dataset.image_files[i]
        base_filename = os.path.splitext(original_filename)[0]
        input_img_denorm = denormalize(inputs_normalized.squeeze().cpu().numpy(), stats['input']['min'], stats['input']['max'])
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Comparison for {original_filename}', fontsize=16)
        axes[0].imshow(input_img_denorm, cmap='gray'); axes[0].set_title('Input Trajectory (Original Scale)'); axes[0].axis('off')
        axes[1].imshow(target_np, cmap='viridis'); axes[1].set_title('Ground Truth Force Field'); axes[1].axis('off')
        axes[2].imshow(predicted_np, cmap='viridis'); axes[2].set_title('Predicted Force Field'); axes[2].axis('off')
        save_path = os.path.join(VISUAL_OUTPUT_DIR, f"comparison_{base_filename}.png")
        plt.savefig(save_path, bbox_inches='tight'); plt.close(fig)

# --- 5. 輸出評估報告 ---
print("\n\n" + "="*25 + " U-Net 模型量化評估報告 " + "="*25)
print(f"\n[ 點對點誤差 (共 {len(test_loader)} 個樣本) ]")
print(f"  - 平均絕對誤差 (MAE): {np.mean(metrics['mae']):.6f} (標準差: {np.std(metrics['mae']):.6f})")
print(f"\n[ 統計特性對比 (平均值 ± 標準差) ]")
print(f"  - 均方根 (RMS):   真實: {np.mean(metrics['real_rms']):.4f} ± {np.std(metrics['real_rms']):.4f} | 預測: {np.mean(metrics['pred_rms']):.4f} ± {np.std(metrics['pred_rms']):.4f}")
print(f"  - 偏度 (Skewness):真實: {np.mean(metrics['real_skew']):.4f} ± {np.std(metrics['real_skew']):.4f} | 預測: {np.mean(metrics['pred_skew']):.4f} ± {np.std(metrics['pred_skew']):.4f}")
print(f"  - 峰度 (Kurtosis):真實: {np.mean(metrics['real_kurt']):.4f} ± {np.std(metrics['real_kurt']):.4f} | 預測: {np.mean(metrics['pred_kurt']):.4f} ± {np.std(metrics['pred_kurt']):.4f}")
print(f"\n[ 物理特性對比 (平均值 ± 標準差) ]")
print(f"  - 關聯長度(像素): 真實: {np.mean(metrics['real_corr_len']):.2f} ± {np.std(metrics['real_corr_len']):.2f} | 預測: {np.mean(metrics['pred_corr_len']):.2f} ± {np.std(metrics['pred_corr_len']):.2f}")
print("="*80)

# --- 6. 生成並儲存匯總圖表 ---
plt.figure(figsize=(10, 6)); plt.hist(metrics["mae"], bins=20, edgecolor='black'); plt.title("平均絕對誤差 (MAE) 分佈"); plt.xlabel("MAE"); plt.ylabel("樣本數"); plt.grid(True, linestyle='--', alpha=0.6); plt.savefig(os.path.join(OUTPUT_DIR, "mae_distribution.png")); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(18, 5)); stats_data = [("RMS", metrics["real_rms"], metrics["pred_rms"]),("Skewness", metrics["real_skew"], metrics["pred_skew"]),("Kurtosis", metrics["real_kurt"], metrics["pred_kurt"])];
for ax, (title, real_data, pred_data) in zip(axes, stats_data):
    ax.boxplot([real_data, pred_data], labels=['真實', '預測']); ax.set_title(title + " 分佈對比"); ax.grid(True, linestyle='--', alpha=0.6)
plt.suptitle("統計特性對比", fontsize=16); plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(OUTPUT_DIR, "statistics_comparison.png")); plt.close()

avg_real_psd = np.mean([p for p in metrics["real_psd"] if p is not None], axis=0)
avg_pred_psd = np.mean([p for p in metrics["pred_psd"] if p is not None], axis=0)
plt.figure(figsize=(10, 6)); plt.plot(avg_real_psd, label='真實力場 (Ground Truth)', color='blue'); plt.plot(avg_pred_psd, label='預測力場 (Predicted)', color='red', linestyle='--'); plt.title("平均1D功率譜密度 (PSD) 對比"); plt.xlabel("空間頻率"); plt.ylabel("功率"); plt.yscale('log'); plt.legend(); plt.grid(True, which="both", linestyle='--', alpha=0.6); plt.savefig(os.path.join(OUTPUT_DIR, "power_spectrum_comparison.png")); plt.close()

min_len = min(len(p) for p in metrics["real_autocorr"] + metrics["pred_autocorr"])
avg_real_ac = np.mean([p[:min_len] for p in metrics["real_autocorr"]], axis=0)
avg_pred_ac = np.mean([p[:min_len] for p in metrics["pred_autocorr"]], axis=0)
plt.figure(figsize=(10, 6)); plt.plot(avg_real_ac, label='真實力場 (Ground Truth)', color='blue'); plt.plot(avg_pred_ac, label='預測力場 (Predicted)', color='red', linestyle='--'); plt.axhline(1/np.e, color='green', linestyle=':', label='1/e 閾值'); plt.title("平均自相關函數"); plt.xlabel("距離 (像素)"); plt.ylabel("歸一化相關性"); plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.savefig(os.path.join(OUTPUT_DIR, "autocorrelation_comparison.png")); plt.close()

print(f"\n量化指標圖表已儲存至 '{OUTPUT_DIR}' 文件夾。")
print(f"可視化對比圖已儲存至 '{VISUAL_OUTPUT_DIR}' 文件夾。")
print("評估完成。")