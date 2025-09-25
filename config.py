# config.py
import os
import torch

# ==============================================================================
# Paths
# ==============================================================================
# Root directory for the project
PROJECT_DIR = '.'

# Root directory for the dataset
# Use an absolute path for robustness, especially with different execution environments.
DATA_DIR = '/lustre/home/2400011491/data/ai_train_data/data_2000'

# Subdirectories for training and testing data
TRAIN_A_DIR = os.path.join(DATA_DIR, 'trainA')
TRAIN_B_DIR = os.path.join(DATA_DIR, 'trainB')
TEST_A_DIR = os.path.join(DATA_DIR, 'testA')
TEST_B_DIR = os.path.join(DATA_DIR, 'testB')
VALIDATION_A_DIR = os.path.join(DATA_DIR, 'validationA')
VALIDATION_B_DIR = os.path.join(DATA_DIR, 'validationB')

# Path to the file for storing normalization statistics
STATS_FILE = os.path.join(PROJECT_DIR, 'data_stats.json')

# Directory to save model checkpoints
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')
# Path to save the best model
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')

# Directory for evaluation results
EVALUATION_DIR = os.path.join(PROJECT_DIR, 'evaluation_results')

# Directory for prediction results
PREDICTION_DIR = os.path.join(PROJECT_DIR, 'predictions')


# ==============================================================================
# Model Parameters
# ==============================================================================
# Number of input channels for the U-Net (e.g., 1 for grayscale)
MODEL_N_CHANNELS = 1
# Number of output classes for the U-Net (e.g., 1 for the force field)
MODEL_N_CLASSES = 1
# Depth of the U-Net (number of down/up sampling layers)
MODEL_DEPTH = 4
# Number of base channels for the first convolutional layer
MODEL_BASE_CHANNELS = 64
# Whether to use bilinear upsampling. If False, uses ConvTranspose2d.
MODEL_BILINEAR = False


# ==============================================================================
# Training Hyperparameters
# ==============================================================================
# Device to use for training ('cuda' or 'cpu')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Number of training epochs
EPOCHS = 200
# Batch size for training and validation
BATCH_SIZE = 16 # Increased from 4, adjust based on your VRAM
# Learning rate for the Adam optimizer
LEARNING_RATE = 1e-4
# Weight decay for the Adam optimizer
WEIGHT_DECAY = 1e-5
# Image size (height and width) to which all images will be resized
IMG_SIZE = 256
# Number of worker processes for the DataLoader
NUM_WORKERS = 4
# Whether to use gradient checkpointing to save memory
USE_CHECKPOINTING = False


# ==============================================================================
# Loss Function Weights
# ==============================================================================
# Weight for the pixel-wise L1 loss
LAMBDA_PIXEL = 1.0
# Weight for the physics-based spectral correlation loss
LAMBDA_PHYSICS = 0.1


# ==============================================================================
# Early Stopping
# ==============================================================================
# Number of epochs with no improvement after which training will be stopped
PATIENCE = 10


# ==============================================================================
# Evaluation Parameters
# ==============================================================================
# Number of visual samples to save during evaluation. -1 means all.
NUM_VISUAL_SAMPLES = 20
