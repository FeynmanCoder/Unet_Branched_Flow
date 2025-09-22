
import argparse
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
    This version avoids mean subtraction to prevent issues with inverse mapping.
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

def train(args):
    """
    Main training and validation loop.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Data Preparation ---
    stats = get_stats(
        args.stats_file,
        os.path.join(args.data_path, 'trainA'),
        os.path.join(args.data_path, 'trainB')
    )
    
    transform = T.Compose([T.Resize((args.img_size, args.img_size), antialias=True)])
    
    train_dataset = ForceFieldDataset(
        dir_A=os.path.join(args.data_path, 'trainA'),
        dir_B=os.path.join(args.data_path, 'trainB'),
        stats=stats,
        transform=transform
    )
    val_dataset = ForceFieldDataset(
        dir_A=os.path.join(args.data_path, 'testA'),
        dir_B=os.path.join(args.data_path, 'testB'),
        stats=stats,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model, Optimizer, Loss ---
    model = UNet(n_channels=1, n_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)
    criterion_pixel = nn.L1Loss()

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(total=len(train_dataset), desc=f"Epoch {epoch + 1}/{args.epochs}", unit='img') as pbar:
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                
                pixel_loss = criterion_pixel(outputs, targets)
                physics_loss = spectral_correlation_loss(outputs, targets)
                total_loss = args.lambda_pixel * pixel_loss + args.lambda_physics * physics_loss
                
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
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logging.info(f"Validation loss did not improve. Patience: {epochs_no_improve}/{args.patience}")

        if epochs_no_improve >= args.patience:
            logging.info("Early stopping triggered.")
            break
            
    logging.info("Training finished.")

def main():
    parser = argparse.ArgumentParser(description="Train U-Net for Force Field Prediction")
    
    # Paths
    parser.add_argument('--data-path', type=str, default='/lustre/home/2400011491/data/ai_train_data/data_20000', help='Root directory of the dataset')
    parser.add_argument('--stats-file', type=str, default='data_stats.json', help='Path to normalization stats file')
    parser.add_argument('--save-path', type=str, default='checkpoints/best_model.pth', help='Path to save the best model')

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=256, help='Image size for resizing')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay for Adam optimizer')
    
    # Loss weights
    parser.add_argument('--lambda-pixel', type=float, default=1.0, help='Weight for pixel-wise L1 loss')
    parser.add_argument('--lambda-physics', type=float, default=0.1, help='Weight for spectral correlation loss')

    # Early stopping
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')

    args = parser.parse_args()
    
    train(args)

if __name__ == '__main__':
    main()
