# predict.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import ForceFieldDataset
from model import UNet

# --- 1. 设置参数 ---
PROJECT_DIR = '.'
DATA_DIR = '/lustre/home/2400011491/data/ai_train_data/data_20000'
TEST_A_DIR = os.path.join(DATA_DIR, 'testA')
TEST_B_DIR = os.path.join(DATA_DIR, 'testB')

# 加载训练好的模型
MODEL_PATH = os.path.join(PROJECT_DIR, "unet_model.pth")
# 创建一个专门的文件夹来存放预测结果
OUTPUT_DIR = os.path.join(PROJECT_DIR, "predictions")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. 加载模型和数据 ---
print("Loading model and data...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = UNet(n_channels=1, n_classes=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
model.eval()

test_dataset = ForceFieldDataset(dir_A=TEST_A_DIR, dir_B=TEST_B_DIR)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) 

print("Starting prediction...")

# --- 3. 进行预测并保存结果 ---
with torch.no_grad():
    for i, (inputs, targets) in enumerate(test_loader):
        # 从数据集中获取原始文件名
        original_filename = test_dataset.image_files[i]
        base_filename = os.path.splitext(original_filename)[0]

        inputs = inputs.to(DEVICE)
        predicted = model(inputs)
        
        input_img = inputs.squeeze().cpu().numpy()
        target_img = targets.squeeze().cpu().numpy()
        predicted_img = predicted.squeeze().cpu().numpy()
        
        # --- 4. 可视化对比 ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Prediction for {original_filename}', fontsize=16)

        axes[0].imshow(input_img, cmap='gray')
        axes[0].set_title('Input Trajectory')
        axes[0].axis('off')

        axes[1].imshow(target_img, cmap='viridis')
        axes[1].set_title('Ground Truth Force Field')
        axes[1].axis('off')

        axes[2].imshow(predicted_img, cmap='viridis')
        axes[2].set_title('Predicted Force Field (U-Net)')
        axes[2].axis('off')
        
        # 使用原始文件名保存图像，更易于追溯
        save_path = os.path.join(OUTPUT_DIR, f"prediction_{base_filename}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved prediction for {original_filename} to {save_path}")

print("Prediction finished.")