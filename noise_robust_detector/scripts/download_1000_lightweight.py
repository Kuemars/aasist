import os
import sys
import librosa
import soundfile as sf
import numpy as np
import random
from tqdm import tqdm
import shutil

# Configuration
TARGET_SR = 16000
TARGET_DURATION = 4.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)
OUTPUT_DIR = "data/raw/clean_real"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set cache to temp location that auto-clears
os.environ['HF_DATASETS_CACHE'] = './temp_hf_cache'

def get_current_count():
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('real_') and f.endswith('.wav')]
    return len(existing)

def clean_hf_cache():
    """Clean Hugging Face cache after each dataset"""
    cache_path = './temp_hf_cache'
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
        os.makedirs(cache_path)

def download_small_dataset(dataset_name, config, split, num_samples=100):
    """Download from smaller, more manageable datasets"""
    print(f"🎯 Downloading {dataset_name}...")
    
    try:
        from datasets import load_dataset
        
        clean_hf_cache()  # Clear cache before starting
        
        # Load with streaming - minimal disk usage
        dataset = load_dataset(dataset_name, config, split=split, streaming=True, trust_remote_code=True)
        
        start_idx = get_current_count() + 1
        saved_count = 0
        
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
                
            try:
                # Get audio data
                if 'audio' in sample and 'array' in sample['audio']:
                    audio_array = sample['audio']['array']
                    sr = sample['audio']['sampling_rate']
                elif 'file' in sample:
                    # Load from file path
                    audio_array, sr = librosa.load(sample['file'], sr=TARGET_SR)
                else:
                    continue
                
                # Process audio
                if sr != TARGET_SR:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)
                
                if len(audio_array) > TARGET_SAMPLES:
                    audio_array = audio_array[:TARGET_SAMPLES]
                else:
                    padding = TARGET_SAMPLES - len(audio_array)
                    audio_array = np.pad(audio_array, (0, padding))
                
                # Save immediately
                filename = f"real_{start_idx + saved_count:04d}.wav"
                output_path = os.path.join(OUTPUT_DIR, filename)
                sf.write(output_path, audio_array, TARGET_SR)
                saved_count += 1
                
                # Show progress every 10 files
                if saved_count % 10 == 0:
                    print(f"   ✅ Saved {saved_count}/{num_samples}")
                
            except Exception as e:
                print(f"   ❌ Error sample {i}: {e}")
                continue
        
        clean_hf_cache()  # Clear cache after completion
        print(f"✅ Saved {saved_count} {dataset_name} samples")
        return saved_count
        
    except Exception as e:
        print(f"❌ {dataset_name} failed: {e}")
        clean_hf_cache()
        return 0

def main():
    print("🚀 LIGHTWEIGHT - DOWNLOADING 1000 REAL VOICES")
    print("=" * 60)
    print("💡 Using streaming + automatic cache cleanup")
    print("💾 Expected disk usage: < 2GB total")
    
    current_count = get_current_count()
    print(f"📊 Current real voices: {current_count}")
    
    target_total = 1000
    needed = target_total - current_count
    
    if needed <= 0:
        print(f"✅ Already have {current_count} voices")
        return
    
    print(f"🎯 Need {needed} more real voices")
    
    # Use SMALLER, well-tested datasets
    datasets = [
        # Small, reliable datasets
        ("librispeech_asr", "clean", "train.100", min(200, needed)),
        ("timit_asr", None, "train", min(150, needed)),
        ("superb", "asr", "train", min(150, needed)),
        ("common_voice", "en", "train", min(200, needed)),
        ("speech_commands", "v0.02", "train", min(100, needed)),
        ("google/fleurs", "en_us", "train", min(100, needed)),
        ("facebook/voxpopuli", "en", "train", min(100, needed)),
    ]
    
    total_downloaded = 0
    remaining = needed
    
    for dataset_name, config, split, target_count in datasets:
        if remaining <= 0:
            break
            
        actual_target = min(target_count, remaining)
        downloaded = download_small_dataset(dataset_name, config, split, actual_target)
        total_downloaded += downloaded
        remaining = needed - total_downloaded
        
        if remaining > 0:
            print(f"📈 Still need {remaining} more voices")
    
    # Final cleanup
    clean_hf_cache()
    if os.path.exists('./temp_hf_cache'):
        shutil.rmtree('./temp_hf_cache')
    
    # Final count
    final_count = get_current_count()
    print(f"\n🎉 DOWNLOAD COMPLETE!")
    print(f"📊 Total real voices: {final_count}")
    print(f"📈 Added: {total_downloaded} new voices")
    
    if final_count < target_total:
        print(f"⚠️  Short by {target_total - final_count} voices")
        print("💡 Run the script again to get more")

if __name__ == "__main__":
    # Create temp cache
    os.makedirs('./temp_hf_cache', exist_ok=True)
    
    try:
        main()
    finally:
        # Always clean up temp cache
        if os.path.exists('./temp_hf_cache'):
            shutil.rmtree('./temp_hf_cache')