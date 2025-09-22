# train.py (最終版 - 根治反向映射問題)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T

from dataset import ForceFieldDataset
from model import UNet

# --- 1. 輔助函數 ---

def get_or_compute_stats(stats_path, train_a_dir, train_b_dir):
    if os.path.exists(stats_path):
        print(f"成功加載已存在的統計文件: '{stats_path}'")
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        return stats
    
    print(f"統計文件 '{stats_path}' 不存在。開始從訓練數據計算...")
    
    def _compute(data_dir):
        global_min = np.finfo(np.float32).max
        global_max = np.finfo(np.float32).min
        file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        for filename in tqdm(file_list, desc=f"處理 {os.path.basename(data_dir)}"):
            path = os.path.join(data_dir, filename)
            data = np.load(path)
            global_min = min(np.min(data), global_min)
            global_max = max(np.max(data), global_max)
        return float(global_min), float(global_max)

    stats_B = _compute(train_b_dir)
    stats_A = _compute(train_a_dir)
    stats = {'input': {'min': stats_B[0], 'max': stats_B[1]}, 'target': {'min': stats_A[0], 'max': stats_A[1]}}
    with open(stats_path, 'w') as f: json.dump(stats, f, indent=4)
    print(f"\n統計信息計算完成並已保存至 '{stats_path}'")
    return stats

# --- 【核心修正】修正物理損失函數 ---
def spectral_correlation_loss(pred_img, target_img):
    """
    通過比較完整的功率譜密度(PSD)來計算物理結構上的損失。
    不再減去平均值，以解決反向映射的對稱性漏洞。
    """
    # 直接對歸一化後的圖像進行傅立葉變換
    pred_fft = torch.fft.fft2(pred_img)
    target_fft = torch.fft.fft2(target_img)
    
    # 計算功率譜密度
    pred_psd = torch.abs(pred_fft)**2
    target_psd = torch.abs(target_fft)**2
    
    # 計算PSD之間的均方誤差
    loss = F.mse_loss(pred_psd, target_psd)
    return loss

# --- 2. 設置參數 ---
PROJECT_DIR = '.'
DATA_DIR = '/lustre/home/2400011491/data/data_2000'
TRAIN_A_DIR = os.path.join(DATA_DIR, 'trainA')
TRAIN_B_DIR = os.path.join(DATA_DIR, 'trainB')
VAL_A_DIR = os.path.join(DATA_DIR, 'testA')
VAL_B_DIR = os.path.join(DATA_DIR, 'testB')
STATS_FILE = 'data_stats.json'

TARGET_SIZE = 256
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 200
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "unet_final_corrected_model.pth")

lambda_pixel = 1.0
lambda_physics = 0.1

print(f"Using device: {DEVICE}")
print(f"Model will be saved to: {MODEL_SAVE_PATH}")

# --- 3. 準備數據 ---
stats = get_or_compute_stats(STATS_FILE, TRAIN_A_DIR, TRAIN_B_DIR)
data_transform = T.Compose([T.Resize((TARGET_SIZE, TARGET_SIZE), antialias=True)])
val_transform = T.Compose([T.Resize((TARGET_SIZE, TARGET_SIZE), antialias=True)])

train_dataset = ForceFieldDataset(dir_A=TRAIN_A_DIR, dir_B=TRAIN_B_DIR, stats=stats, transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = ForceFieldDataset(dir_A=VAL_A_DIR, dir_B=VAL_B_DIR, stats=stats, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. 初始化模型、損失函数和優化器 ---
model = UNet(n_channels=1, n_classes=1).to(DEVICE)
criterion_pixel = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# --- 5. 訓練循環 (現在是正確的邏輯) ---
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    loop = tqdm(train_loader, leave=True)
    loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
    
    for inputs, targets in loop:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        
        # 使用修正後的物理損失
        pixel_loss = criterion_pixel(outputs, targets)
        physics_loss = spectral_correlation_loss(outputs, targets)
        total_loss = lambda_pixel * pixel_loss + lambda_physics * physics_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        loop.set_postfix(pixel_loss=pixel_loss.item(), physics_loss=physics_loss.item())

    # 驗證模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion_pixel(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Avg Validation L1 Loss (normalized): {avg_val_loss:.6f}")
    
    # Early Stopping 邏輯
    if avg_val_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss:.6f} to {avg_val_loss:.6f}. Saving model...")
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"Validation loss did not improve. Patience: {epochs_no_improve}/{EARLY_STOP_PATIENCE}")

    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print("Early stopping triggered. Training finished.")
        break

print("Training finished.")