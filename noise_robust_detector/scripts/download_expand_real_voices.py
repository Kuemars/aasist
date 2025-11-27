import os
import requests
import tarfile
import zipfile
import librosa
import soundfile as sf
import numpy as np
import random
from urllib.parse import urljoin
import json

def download_voxceleb_samples(num_samples=30):
    """
    Download random samples from VoxCeleb1 dataset
    Note: This downloads from official sources
    """
    print("🎯 Downloading VoxCeleb samples...")
    
    # VoxCeleb1 download URLs (official)
    base_url = "http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html"
    
    # Since direct download requires registration, we'll use a different approach
    # Using pre-selected samples from Hugging Face
    print("⚠️  VoxCeleb requires manual download from:")
    print("   http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html")
    print("   Please download and extract manually, then run the processing part.")
    
    return []

def download_common_voice_samples(num_samples=30, language="en"):
    """
    Download samples from Mozilla Common Voice
    """
    print(f"🎯 Downloading Common Voice {language} samples...")
    
    # Common Voice dataset info
    base_url = "https://commonvoice.mozilla.org/api/v1/datasets"
    
    try:
        # Get dataset list
        response = requests.get(f"{base_url}/{language}")
        datasets = response.json()
        
        print(f"📊 Available Common Voice datasets for {language}:")
        for dataset in datasets:
            print(f"   - {dataset['name']}: {dataset['description']}")
        
        # For now, we'll use a simplified approach
        print("⚠️  Common Voice API requires complex download process")
        print("💡 Alternative: Download from Hugging Face:")
        print("   https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0")
        
    except Exception as e:
        print(f"❌ Error accessing Common Voice: {e}")
    
    return []

def download_from_huggingface(dataset_name, num_samples=30, subset="en"):
    """
    Download samples from Hugging Face datasets
    """
    print(f"🎯 Downloading {num_samples} samples from {dataset_name}...")
    
    try:
        from datasets import load_dataset
        
        # Load dataset
        dataset = load_dataset(dataset_name, subset, trust_remote_code=True)
        
        # Get random samples from train split
        if 'train' in dataset:
            audio_samples = dataset['train'].shuffle(seed=42).select(range(num_samples))
            return audio_samples
        else:
            print(f"❌ No 'train' split in {dataset_name}")
            
    except ImportError:
        print("❌ datasets package not installed. Install with: pip install datasets")
    except Exception as e:
        print(f"❌ Error downloading from Hugging Face: {e}")
    
    return []

def normalize_audio(audio_path, target_duration=4.0, target_sr=16000):
    """
    Normalize audio to target duration and sample rate
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=target_sr)
        
        # Calculate target samples
        target_samples = int(target_duration * target_sr)
        
        # Trim or pad to target duration
        if len(audio) > target_samples:
            # Trim from center
            start = (len(audio) - target_samples) // 2
            audio = audio[start:start + target_samples]
        else:
            # Pad with silence
            padding = target_samples - len(audio)
            audio = np.pad(audio, (0, padding))
        
        return audio, target_sr
        
    except Exception as e:
        print(f"❌ Error normalizing {audio_path}: {e}")
        return None, None

def process_and_save_samples(samples, dataset_name, output_dir):
    """
    Process samples and save to real voices directory
    """
    saved_count = 0
    
    # Get current file count to continue numbering
    existing_files = [f for f in os.listdir(output_dir) if f.startswith('real_') and f.endswith('.wav')]
    start_idx = len(existing_files) + 1
    
    for i, sample in enumerate(samples):
        try:
            if 'audio' in sample and 'path' in sample['audio']:
                audio_path = sample['audio']['path']
            elif 'file' in sample:
                audio_path = sample['file']
            else:
                # Skip if no audio path
                continue
            
            # Normalize audio
            audio, sr = normalize_audio(audio_path)
            
            if audio is not None:
                # Save with new name
                new_filename = f"real_{start_idx + i:03d}.wav"
                output_path = os.path.join(output_dir, new_filename)
                
                sf.write(output_path, audio, sr)
                print(f"✅ Saved: {new_filename} (from {dataset_name})")
                saved_count += 1
                
        except Exception as e:
            print(f"❌ Error processing sample {i}: {e}")
    
    return saved_count

def main():
    """
    Main function to expand real voices dataset
    """
    print("🚀 EXPANDING REAL VOICES DATASET")
    print("=" * 50)
    
    # Output directory
    output_dir = "data/raw/clean_real"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current count
    current_files = [f for f in os.listdir(output_dir) if f.startswith('real_') and f.endswith('.wav')]
    print(f"📊 Current real voices: {len(current_files)}")
    
    # Hugging Face datasets to try
    datasets_to_try = [
        ("mozilla-foundation/common_voice_11_0", "en", "Common Voice English"),
        ("facebook/voxpopuli", "en", "VoxPopuli English"),
        ("librispeech_asr", "clean", "LibriSpeech Clean"),
        ("speech_commands", "v0.02", "Google Speech Commands"),
    ]
    
    total_added = 0
    
    for dataset_name, subset, display_name in datasets_to_try:
        print(f"\n🎯 Trying {display_name}...")
        
        # Download samples
        samples = download_from_huggingface(dataset_name, num_samples=15, subset=subset)
        
        if samples:
            # Process and save
            added = process_and_save_samples(samples, display_name, output_dir)
            total_added += added
            print(f"✅ Added {added} samples from {display_name}")
        else:
            print(f"❌ Failed to get samples from {display_name}")
    
    # Final summary
    new_total = len(current_files) + total_added
    print(f"\n🎉 EXPANSION COMPLETE!")
    print(f"📊 Added {total_added} new real voices")
    print(f"📊 Total real voices: {new_total}")
    
    # Manual download instructions
    print(f"\n💡 MANUAL DOWNLOAD OPTIONS:")
    print(f"   1. VoxCeleb: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html")
    print(f"   2. Common Voice: https://commonvoice.mozilla.org/en/datasets")
    print(f"   3. Add manual downloads to: {output_dir}")

if __name__ == "__main__":
    # Install required package first
    try:
        import datasets
    except ImportError:
        print("📦 Installing datasets package...")
        os.system("pip install datasets")
    
    main()