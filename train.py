
import os
import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import all settings from the config file
import config
from dataset import ForceFieldDataset
from model import UNet

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_stats(stats_path, train_a_dir, train_b_dir):
    """
    Load or compute normalization statistics for the dataset.
    """
    if os.path.exists(stats_path):
        logging.info(f"Loading existing stats from '{stats_path}'")
        with open(stats_path, 'r') as f:
            return json.load(f)
    
    logging.info(f"Stats file not found. Computing from training data...")
    
    def _compute(data_dir):
        all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
        
        min_val = np.finfo(np.float32).max
        max_val = np.finfo(np.float32).min
        
        for f in tqdm(all_files, desc=f"Analyzing {os.path.basename(data_dir)}"):
            data = np.load(f)
            min_val = min(np.min(data), min_val)
            max_val = max(np.max(data), max_val)
            
        return float(min_val), float(max_val)

    stats_A = _compute(train_a_dir)
    stats_B = _compute(train_b_dir)
    
    stats = {
        'input': {'min': stats_B[0], 'max': stats_B[1]},
        'target': {'min': stats_A[0], 'max': stats_A[1]}
    }
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    logging.info(f"Stats computed and saved to '{stats_path}'")
    
    return stats

def spectral_correlation_loss(pred, target):
    """
    Calculate physics-based loss by comparing the Power Spectral Density (PSD).
    """
    pred_fft = torch.fft.fft2(pred)
    target_fft = torch.fft.fft2(target)
    
    pred_psd = torch.abs(pred_fft)**2
    target_psd = torch.abs(target_fft)**2
    
    return F.mse_loss(pred_psd, target_psd)

def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def train():
    """
    Main training and validation loop.
    """
    device = torch.device(config.DEVICE)
    logging.info(f"Using device: {device}")

    # --- Data Preparation ---
    stats = get_stats(config.STATS_FILE, config.TRAIN_A_DIR, config.TRAIN_B_DIR)
    
    transform = T.Compose([T.Resize((config.IMG_SIZE, config.IMG_SIZE), antialias=True)])
    
    train_dataset = ForceFieldDataset(
        input_dir=config.TRAIN_B_DIR,
        target_dir=config.TRAIN_A_DIR,
        stats=stats,
        transform=transform
    )
    val_dataset = ForceFieldDataset(
        input_dir=config.VALIDATION_B_DIR,
        target_dir=config.VALIDATION_A_DIR,
        stats=stats,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=True
    )

    # --- Model, Optimizer, Loss ---
    model = UNet(
        n_channels=config.MODEL_N_CHANNELS,
        n_classes=config.MODEL_N_CLASSES,
        bilinear=config.MODEL_BILINEAR,
        depth=config.MODEL_DEPTH,
        base_channels=config.MODEL_BASE_CHANNELS
    ).to(device)

    if config.USE_CHECKPOINTING:
        model.use_checkpointing()
        logging.info("Gradient checkpointing enabled.")

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.1
    )
    criterion_pixel = nn.L1Loss()

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0
        
        with tqdm(total=len(train_dataset), desc=f"Epoch {epoch + 1}/{config.EPOCHS}", unit='img') as pbar:
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                
                pixel_loss = criterion_pixel(outputs, targets)
                physics_loss = spectral_correlation_loss(outputs, targets)
                total_loss = config.LAMBDA_PIXEL * pixel_loss + config.LAMBDA_PHYSICS * physics_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                pbar.update(inputs.size(0))
                epoch_loss += total_loss.item()
                pbar.set_postfix(**{'loss (batch)': total_loss.item()})

        # --- Validation ---
        avg_val_loss = evaluate(model, val_loader, criterion_pixel, device)
        logging.info(f"Validation L1 Loss: {avg_val_loss:.6f}")
        
        scheduler.step(avg_val_loss)

        # --- Save Model & Early Stopping ---
        if avg_val_loss < best_val_loss:
            logging.info(f"Validation loss improved from {best_val_loss:.6f} to {avg_val_loss:.6f}. Saving model...")
            best_val_loss = avg_val_loss
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logging.info(f"Validation loss did not improve. Patience: {epochs_no_improve}/{config.PATIENCE}")

        if epochs_no_improve >= config.PATIENCE:
            logging.info("Early stopping triggered.")
            break
            
    logging.info("Training finished.")

if __name__ == '__main__':
    train()
