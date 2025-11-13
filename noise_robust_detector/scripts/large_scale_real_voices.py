"""
LARGE-SCALE REAL VOICE COLLECTION - Open Source Datasets
"""

import os
import sys

# Fix import path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from project_config import *

def get_large_scale_datasets():
    """Identify best large-scale open source datasets"""
    
    print("🎯 LARGE-SCALE OPEN SOURCE DATASETS")
    print("=" * 50)
    
    datasets = {
        "LibriSpeech": {
            "size": "1,000 hours",
            "speakers": "2,484 speakers",
            "diversity": "Professional, balanced gender, clear audio",
            "download_url": "http://www.openslr.org/12/",
            "recommended_subset": "train-clean-100 (100 hours)",
            "samples_needed": 200,
            "use_case": "Primary training - high quality"
        },
        "Common Voice": {
            "size": "10,000+ hours", 
            "speakers": "50,000+ speakers",
            "diversity": "Global accents, ages, backgrounds, real-world",
            "download_url": "https://commonvoice.mozilla.org/en/datasets",
            "recommended_subset": "English samples",
            "samples_needed": 300,
            "use_case": "Accent and background diversity"
        },
        "VoxCeleb1": {
            "size": "100 hours", 
            "speakers": "1,251 speakers",
            "diversity": "Celebrities, interviews, real noise conditions",
            "download_url": "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/",
            "recommended_subset": "Full dataset",
            "samples_needed": 100,
            "use_case": "Robustness to real-world conditions"
        }
    }
    
    total_samples = 0
    for name, info in datasets.items():
        print(f"\n🎯 {name}:")
        print(f"   Size: {info['size']}")
        print(f"   Speakers: {info['speakers']}")
        print(f"   Diversity: {info['diversity']}")
        print(f"   Samples: {info['samples_needed']} voices")
        print(f"   Download: {info['download_url']}")
        total_samples += info['samples_needed']
    
    print(f"\n📊 TOTAL TARGET: {total_samples} diverse real voices")
    return datasets, total_samples

def create_download_plan():
    """Create step-by-step download plan"""
    
    print("\n📥 DOWNLOAD EXECUTION PLAN:")
    
    steps = [
        ("1. LibriSpeech", "Download train-clean-100.tar.gz", "~6.3GB", "Primary quality data"),
        ("2. Common Voice", "Download English dataset", "~15GB", "Maximum diversity"),
        ("3. VoxCeleb1", "Download full dataset", "~8GB", "Real-world robustness"),
        ("4. Extract & Filter", "Keep only clean speech samples", "-", "Quality control"),
        ("5. Verify Diversity", "Check gender/age/accent balance", "-", "Training readiness")
    ]
    
    for step, action, size, purpose in steps:
        print(f"   {step:20} {action:30} {size:10} {purpose}")

def setup_download_scripts():
    """Create automated download scripts"""
    
    print("\n🔧 CREATING DOWNLOAD SCRIPTS...")
    
    # Create download script for LibriSpeech
    librispeech_script = """
#!/bin/bash
# download_librispeech.sh

echo "Downloading LibriSpeech train-clean-100..."
wget http://www.openslr.org/resources/12/train-clean-100.tar.gz

echo "Extracting..."
tar -xzf train-clean-100.tar.gz

echo "Converting FLAC to WAV and copying to project..."
python scripts/process_librispeech.py
"""
    
    # Create processing script
    processing_script = """
# process_librispeech.py
import os
import librosa
import soundfile as sf
from project_config import DATA_RAW

def process_librispeech():
    \"\"\"Convert LibriSpeech to our format\"\"\"
    print("Processing LibriSpeech samples...")
    
    count = 0
    for root, dirs, files in os.walk("LibriSpeech/train-clean-100"):
        for file in files:
            if file.endswith('.flac') and count < 200:
                # Convert to WAV and copy to project
                input_path = os.path.join(root, file)
                output_path = os.path.join(DATA_RAW, "clean_real", f"librispeech_{count:04d}.wav")
                
                audio, sr = librosa.load(input_path, sr=16000)
                sf.write(output_path, audio, sr)
                count += 1
                
                if count % 50 == 0:
                    print(f"Processed {count} samples...")
    
    print(f"✅ Processed {count} LibriSpeech samples")

if __name__ == "__main__":
    process_librispeech()
"""
    
    print("✅ Download scripts will be created")
    print("💡 These will automate the data collection process")

def main():
    print("🎯 STEP 3: LARGE-SCALE REAL VOICE COLLECTION")
    print("=" * 60)
    
    # Get dataset information
    datasets, total_samples = get_large_scale_datasets()
    
    # Create download plan
    create_download_plan()
    
    # Setup download scripts
    setup_download_scripts()
    
    print("\n" + "=" * 60)
    print(f"🎯 TARGET: {total_samples} diverse real voices")
    print("💡 Next: Download LibriSpeech (start with highest quality)")
    print("   This will give you 200+ professional voice samples")
    print("=" * 60)

if __name__ == "__main__":
    main()