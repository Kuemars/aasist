"""
NOISE-ROBUST AI VOICE DETECTOR - PROJECT CONFIGURATION
Version: 1.0
Objective: Train custom model to detect AI voices through background noise
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed") 
DATA_AUGMENTED = os.path.join(PROJECT_ROOT, "data", "augmented")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Audio parameters
SAMPLE_RATE = 16000
AUDIO_LENGTH = 64600  # ~4 seconds at 16kHz
N_MELS = 64           # Mel spectrogram dimensions

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Noise augmentation parameters
NOISE_TYPES = ["white", "office", "traffic", "keyboard", "cafe"]
NOISE_LEVELS = [0.01, 0.05, 0.1]  # Noise-to-signal ratio

def verify_structure():
    """Verify all project directories exist"""
    required_dirs = [
        DATA_RAW, DATA_PROCESSED, DATA_AUGMENTED,
        os.path.join(DATA_RAW, "clean_real"),
        os.path.join(DATA_RAW, "clean_ai"), 
        os.path.join(DATA_RAW, "noise_samples"),
        MODELS_DIR, SCRIPTS_DIR, RESULTS_DIR
    ]
    
    print("🔍 Verifying project structure...")
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created: {directory}")
        else:
            print(f"✅ Found: {directory}")
    
    print("🎯 Project structure ready!")

if __name__ == "__main__":
    verify_structure()
