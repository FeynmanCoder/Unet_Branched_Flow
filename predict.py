# predict.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as T
import json
from tqdm import tqdm

# Import all settings from the config file
import config
from dataset import ForceFieldDataset
from model import UNet

def denormalize(data, min_val, max_val):
    """Denormalize data from [0, 1] range to original scale."""
    # Ensure data is a numpy array on the CPU
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    # Handle cases where min and max are the same
    if (max_val - min_val) == 0:
        return data
    return data * (max_val - min_val) + min_val

def predict():
    """Main prediction function."""
    # --- 1. Setup from Config ---
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    os.makedirs(config.PREDICTION_DIR, exist_ok=True)

    # --- 2. Load Model and Data ---
    print("Loading model and data...")

    # Load normalization statistics to correctly denormalize the output
    try:
        with open(config.STATS_FILE, 'r') as f:
            stats = json.load(f)
        print(f"Successfully loaded stats from '{config.STATS_FILE}'")
    except FileNotFoundError:
        print(f"Warning: Stats file not found at '{config.STATS_FILE}'. Predictions will be shown in normalized [0, 1] scale.")
        # Create a dummy stats dict to allow the script to run
        stats = {
            'input': {'min': 0, 'max': 1},
            'target': {'min': 0, 'max': 1}
        }

    # Load model
    model = UNet(
        n_channels=config.MODEL_N_CHANNELS,
        n_classes=config.MODEL_N_CLASSES,
        bilinear=config.MODEL_BILINEAR,
        depth=config.MODEL_DEPTH,
        base_channels=config.MODEL_BASE_CHANNELS
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
    # Use a larger batch size for faster prediction if VRAM allows
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS) 

    print("Starting prediction...")

    # --- 3. Perform Prediction and Save Results ---
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(test_loader, desc="Predicting")):
            # Get original filenames for this batch
            start_index = i * config.BATCH_SIZE
            end_index = start_index + inputs.size(0)
            original_filenames = test_dataset.image_files[start_index:end_index]

            inputs = inputs.to(device)
            predicted_normalized = model(inputs)
            
            # Process each image in the batch
            for j in range(inputs.size(0)):
                base_filename = os.path.splitext(original_filenames[j])[0]

                # Denormalize images for visualization
                input_img = denormalize(inputs[j].squeeze(), stats['input']['min'], stats['input']['max'])
                target_img = denormalize(targets[j].squeeze(), stats['target']['min'], stats['target']['max'])
                predicted_img = denormalize(predicted_normalized[j].squeeze(), stats['target']['min'], stats['target']['max'])
                
                # --- 4. Visualize and Save ---
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle(f'Prediction for {original_filenames[j]}', fontsize=16)

                axes[0].imshow(input_img, cmap='gray')
                axes[0].set_title('Input Trajectory')
                axes[0].axis('off')

                axes[1].imshow(target_img, cmap='viridis')
                axes[1].set_title('Ground Truth Force Field')
                axes[1].axis('off')

                axes[2].imshow(predicted_img, cmap='viridis')
                axes[2].set_title('Predicted Force Field (U-Net)')
                axes[2].axis('off')
                
                save_path = os.path.join(config.PREDICTION_DIR, f"prediction_{base_filename}.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
        
    print(f"Prediction finished. Results saved to '{config.PREDICTION_DIR}'.")

if __name__ == '__main__':
    predict()