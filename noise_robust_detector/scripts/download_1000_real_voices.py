import os
import sys
import requests
import tarfile
import zipfile
import librosa
import soundfile as sf
import numpy as np
import random
from tqdm import tqdm
from urllib.parse import urljoin
import subprocess

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Configuration
TARGET_SR = 16000
TARGET_DURATION = 4.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)
OUTPUT_DIR = "data/raw/clean_real"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_current_count():
    """Get current number of real voice files"""
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('real_') and f.endswith('.wav')]
    return len(existing)

def normalize_audio(audio_path):
    """Normalize audio to target duration and sample rate"""
    try:
        audio, sr = librosa.load(audio_path, sr=TARGET_SR)
        
        # Trim or pad to target duration
        if len(audio) > TARGET_SAMPLES:
            # Trim from center
            start = (len(audio) - TARGET_SAMPLES) // 2
            audio = audio[start:start + TARGET_SAMPLES]
        else:
            # Pad with silence
            padding = TARGET_SAMPLES - len(audio)
            audio = np.pad(audio, (0, padding))
        
        return audio, TARGET_SR
    except Exception as e:
        print(f"❌ Error normalizing {audio_path}: {e}")
        return None, None

def save_audio(audio, sr, filename):
    """Save audio file with proper naming"""
    output_path = os.path.join(OUTPUT_DIR, filename)
    sf.write(output_path, audio, sr)
    return output_path

def download_librispeech_samples(num_samples=300):
    """Download samples from LibriSpeech"""
    print("🎯 Downloading LibriSpeech samples...")
    
    try:
        from datasets import load_dataset
        
        # Load LibriSpeech
        dataset = load_dataset("librispeech_asr", "clean", split="train", trust_remote_code=True)
        
        # Get random samples
        selected_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        samples = [dataset[i] for i in selected_indices]
        
        start_idx = get_current_count() + 1
        saved_count = 0
        
        for i, sample in enumerate(tqdm(samples, desc="LibriSpeech")):
            try:
                audio_array = sample['audio']['array']
                sr = sample['audio']['sampling_rate']
                
                # Resample if needed
                if sr != TARGET_SR:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)
                
                # Ensure correct length
                if len(audio_array) > TARGET_SAMPLES:
                    audio_array = audio_array[:TARGET_SAMPLES]
                else:
                    padding = TARGET_SAMPLES - len(audio_array)
                    audio_array = np.pad(audio_array, (0, padding))
                
                # Save
                filename = f"real_{start_idx + i:04d}.wav"
                save_audio(audio_array, TARGET_SR, filename)
                saved_count += 1
                
            except Exception as e:
                print(f"❌ Error processing LibriSpeech sample {i}: {e}")
                continue
        
        print(f"✅ Saved {saved_count} LibriSpeech samples")
        return saved_count
        
    except Exception as e:
        print(f"❌ LibriSpeech download failed: {e}")
        return 0

def download_common_voice_samples(num_samples=300):
    """Download samples from Mozilla Common Voice"""
    print("🎯 Downloading Common Voice samples...")
    
    try:
        from datasets import load_dataset
        
        # Load Common Voice English
        dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train", trust_remote_code=True)
        
        # Filter for validated samples
        validated_samples = [s for s in dataset if s['up_votes'] > 0 and s['down_votes'] == 0]
        
        # Get random samples
        selected_samples = random.sample(validated_samples, min(num_samples, len(validated_samples)))
        
        start_idx = get_current_count() + 1
        saved_count = 0
        
        for i, sample in enumerate(tqdm(selected_samples, desc="Common Voice")):
            try:
                audio_path = sample['path']
                audio, sr = normalize_audio(audio_path)
                
                if audio is not None:
                    filename = f"real_{start_idx + i:04d}.wav"
                    save_audio(audio, sr, filename)
                    saved_count += 1
                    
            except Exception as e:
                print(f"❌ Error processing Common Voice sample {i}: {e}")
                continue
        
        print(f"✅ Saved {saved_count} Common Voice samples")
        return saved_count
        
    except Exception as e:
        print(f"❌ Common Voice download failed: {e}")
        return 0

def download_voxceleb_samples(num_samples=300):
    """Download samples from VoxCeleb1"""
    print("🎯 Downloading VoxCeleb samples...")
    
    try:
        from datasets import load_dataset
        
        # Load VoxCeleb1
        dataset = load_dataset("superb", "voxceleb1", split="train", trust_remote_code=True)
        
        # Get random samples
        selected_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        samples = [dataset[i] for i in selected_indices]
        
        start_idx = get_current_count() + 1
        saved_count = 0
        
        for i, sample in enumerate(tqdm(samples, desc="VoxCeleb")):
            try:
                audio_path = sample['file']
                audio, sr = normalize_audio(audio_path)
                
                if audio is not None:
                    filename = f"real_{start_idx + i:04d}.wav"
                    save_audio(audio, sr, filename)
                    saved_count += 1
                    
            except Exception as e:
                print(f"❌ Error processing VoxCeleb sample {i}: {e}")
                continue
        
        print(f"✅ Saved {saved_count} VoxCeleb samples")
        return saved_count
        
    except Exception as e:
        print(f"❌ VoxCeleb download failed: {e}")
        return 0

def download_voxpopuli_samples(num_samples=100):
    """Download samples from VoxPopuli"""
    print("🎯 Downloading VoxPopuli samples...")
    
    try:
        from datasets import load_dataset
        
        # Load VoxPopuli English
        dataset = load_dataset("facebook/voxpopuli", "en", split="train", trust_remote_code=True)
        
        # Get random samples
        selected_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        samples = [dataset[i] for i in selected_indices]
        
        start_idx = get_current_count() + 1
        saved_count = 0
        
        for i, sample in enumerate(tqdm(samples, desc="VoxPopuli")):
            try:
                audio_array = sample['audio']['array']
                sr = sample['audio']['sampling_rate']
                
                # Resample if needed
                if sr != TARGET_SR:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)
                
                # Ensure correct length
                if len(audio_array) > TARGET_SAMPLES:
                    audio_array = audio_array[:TARGET_SAMPLES]
                else:
                    padding = TARGET_SAMPLES - len(audio_array)
                    audio_array = np.pad(audio_array, (0, padding))
                
                # Save
                filename = f"real_{start_idx + i:04d}.wav"
                save_audio(audio_array, TARGET_SR, filename)
                saved_count += 1
                
            except Exception as e:
                print(f"❌ Error processing VoxPopuli sample {i}: {e}")
                continue
        
        print(f"✅ Saved {saved_count} VoxPopuli samples")
        return saved_count
        
    except Exception as e:
        print(f"❌ VoxPopuli download failed: {e}")
        return 0

def main():
    print("🚀 DOWNLOADING 1000 REAL VOICES")
    print("=" * 60)
    
    current_count = get_current_count()
    print(f"📊 Current real voices: {current_count}")
    
    target_total = 1000
    needed = target_total - current_count
    
    if needed <= 0:
        print(f"✅ Already have {current_count} voices (target: {target_total})")
        return
    
    print(f"🎯 Need {needed} more real voices")
    
    # Distribution for 1000 voices
    targets = {
        "LibriSpeech": min(300, needed // 4),
        "Common Voice": min(300, needed // 4), 
        "VoxCeleb": min(300, needed // 4),
        "VoxPopuli": min(100, needed // 4)
    }
    
    total_downloaded = 0
    
    for source, target_count in targets.items():
        if total_downloaded >= needed:
            break
            
        if target_count > 0:
            print(f"\n📥 Downloading from {source}...")
            
            if source == "LibriSpeech":
                downloaded = download_librispeech_samples(target_count)
            elif source == "Common Voice":
                downloaded = download_common_voice_samples(target_count)
            elif source == "VoxCeleb":
                downloaded = download_voxceleb_samples(target_count)
            elif source == "VoxPopuli":
                downloaded = download_voxpopuli_samples(target_count)
            
            total_downloaded += downloaded
    
    # Final count
    final_count = get_current_count()
    print(f"\n🎉 DOWNLOAD COMPLETE!")
    print(f"📊 Total real voices: {final_count}")
    print(f"📈 Added: {total_downloaded} new voices")
    
    if final_count < target_total:
        print(f"⚠️  Short by {target_total - final_count} voices")
        print("💡 Consider running again or adding manual datasets")

if __name__ == "__main__":
    # Install datasets package if not available
    try:
        import datasets
    except ImportError:
        print("📦 Installing datasets package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    
    main()