# scripts/project_config_extended.py
import os
from pathlib import Path

# Import your existing config
import project_config

# Extended configuration that works with your structure
class ExtendedConfig:
    def __init__(self):
        # Use your existing paths
        self.PROJECT_ROOT = project_config.PROJECT_ROOT
        self.DATA_RAW = project_config.DATA_RAW
        self.DATA_PROCESSED = project_config.DATA_PROCESSED
        self.DATA_AUGMENTED = project_config.DATA_AUGMENTED
        self.MODELS_DIR = project_config.MODELS_DIR
        self.SCRIPTS_DIR = project_config.SCRIPTS_DIR
        self.RESULTS_DIR = project_config.RESULTS_DIR
        
        # Audio parameters from your config
        self.SAMPLE_RATE = project_config.SAMPLE_RATE
        self.AUDIO_LENGTH = project_config.AUDIO_LENGTH
        self.TARGET_DURATION = 4.04  # seconds
        
        # Specific directories
        self.REAL_RAW_DIR = Path(self.DATA_RAW) / "clean_real"
        self.AI_RAW_DIR = Path(self.DATA_RAW) / "clean_ai"
        self.NOISE_RAW_DIR = Path(self.DATA_RAW) / "noise_samples"
        
        # New dataset directories
        self.ELEVEN_LABS_RAW_DIR = Path(self.DATA_RAW) / "eleven_labs"
        self.ASVSPOOF_RAW_DIR = Path(self.DATA_RAW) / "asvspoof"
        self.MEDELEY_RAW_DIR = Path(self.DATA_RAW) / "medeley"
        
        # Dataset URLs
        self.AI_DATASET_URLS = {
            "eleven_labs": "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/79g59sp69z-1.zip",
            "asvspoof": "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip",
            "fake_or_real": "https://github.com/jonasvdd/TTSAudioDeepfakeDetection/releases/download/v1.0.0/train.zip"
        }

# Create global config instance
config = ExtendedConfig()