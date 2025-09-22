# dataset.py (最終版 - 支持歸一化)

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ForceFieldDataset(Dataset):
    """
    自定義數據集，用於加載成對的力場和軌跡圖像。
    增加了對圖像轉換和歸一化的支持。
    """
    def __init__(self, dir_A, dir_B, stats, transform=None):
        """
        Args:
            dir_A (string): 存放力場圖像 (trainA) 的文件夾路徑。
            dir_B (string): 存放軌跡圖像 (trainB) 的文件夾路徑。
            stats (dict): 包含 'input' 和 'target' 的 min/max 值的字典。
            transform (callable, optional): 應用於樣本的可選轉換。
        """
        self.dir_A = dir_A
        self.dir_B = dir_B
        self.image_files = sorted([f for f in os.listdir(dir_B) if f.endswith('.npy')])
        self.transform = transform
        
        # 保存傳入的統計數據
        self.input_min = stats['input']['min']
        self.input_max = stats['input']['max']
        self.target_min = stats['target']['min']
        self.target_max = stats['target']['max']

    def __len__(self):
        return len(self.image_files)
    
    def normalize(self, data, min_val, max_val):
        """將數據歸一化到 [0, 1] 範圍。"""
        # 避免除以零
        if (max_val - min_val) == 0:
            return data
        return (data - min_val) / (max_val - min_val)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        path_A = os.path.join(self.dir_A, img_name)
        path_B = os.path.join(self.dir_B, img_name)
        
        force_field_np = np.load(path_A)
        trajectory_np = np.load(path_B)

        # 應用歸一化
        trajectory_normalized = self.normalize(trajectory_np, self.input_min, self.input_max)
        force_field_normalized = self.normalize(force_field_np, self.target_min, self.target_max)

        # 轉換為張量
        trajectory_tensor = torch.from_numpy(trajectory_normalized).float().unsqueeze(0)
        force_field_tensor = torch.from_numpy(force_field_normalized).float().unsqueeze(0)
        
        if self.transform:
            trajectory_tensor = self.transform(trajectory_tensor)
            force_field_tensor = self.transform(force_field_tensor)
        
        return trajectory_tensor, force_field_tensor