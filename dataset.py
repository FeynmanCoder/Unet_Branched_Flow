# dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ForceFieldDataset(Dataset):
    """
    Custom dataset for loading paired force field (target) and trajectory (input) images.
    Supports normalization and data augmentation transforms.
    """
    def __init__(self, input_dir, target_dir, stats, transform=None):
        """
        Args:
            input_dir (string): Directory with input images (e.g., trainB).
            target_dir (string): Directory with target images (e.g., trainA).
            stats (dict): A dictionary containing min/max values for 'input' and 'target'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.image_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
        self.transform = transform
        
        # Store normalization statistics
        self.input_min = stats['input']['min']
        self.input_max = stats['input']['max']
        self.target_min = stats['target']['min']
        self.target_max = stats['target']['max']

    def __len__(self):
        return len(self.image_files)
    
    def normalize(self, data, min_val, max_val):
        """Normalize data to the [0, 1] range."""
        # Avoid division by zero
        if (max_val - min_val) == 0:
            return data
        return (data - min_val) / (max_val - min_val)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        input_path = os.path.join(self.input_dir, img_name)
        target_path = os.path.join(self.target_dir, img_name)
        
        input_np = np.load(input_path)
        target_np = np.load(target_path)

        # Apply normalization
        input_normalized = self.normalize(input_np, self.input_min, self.input_max)
        target_normalized = self.normalize(target_np, self.target_min, self.target_max)

        # Convert to tensors
        input_tensor = torch.from_numpy(input_normalized).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_normalized).float().unsqueeze(0)
        
        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)
        
        return input_tensor, target_tensor