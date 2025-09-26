# evaluate.py (Refactored Version)

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