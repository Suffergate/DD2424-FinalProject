import torch
import os

# Data settings
TRAIN_DATA_PATH = 'Datasets/bob_dylan_train.txt'
VAL_DATA_PATH = 'Datasets/bob_dylan_val.txt'
TEST_DATA_PATH = 'Datasets/bob_dylan_test.txt'
PROCESSED_DATA_PATH = 'data/processed_data.pkl'

MODEL_SAVE_DIR = 'results/models'
SAMPLE_SAVE_DIR = 'results/samples'
ANALYSIS_SAVE_DIR = 'results/analysis'
PLOT_SAVE_DIR = 'results/plots'
DATA_PATH = 'Datasets/bob_dylan_clean.txt'

# Model parameters
SEQ_LENGTH = 100
BATCH_SIZE = 64
VOCAB_SIZE = None 
EMBEDDING_DIM = 256
HIDDEN_DIM = 128
DROPOUT = 0.2

# Training settings
LEARNING_RATE = 0.001
NUM_EPOCHS = 1

# Generation settings
DEFAULT_TEMPERATURE = 1.0
DEFAULT_NUCLEUS_P = 0.9

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create directories if they don't exist
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
os.makedirs(SAMPLE_SAVE_DIR, exist_ok=True)
os.makedirs(ANALYSIS_SAVE_DIR, exist_ok=True)