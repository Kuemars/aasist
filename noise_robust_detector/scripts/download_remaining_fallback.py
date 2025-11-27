import os
import sys
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Configuration
TARGET_SR = 16000
TARGET_DURATION = 4.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)
OUTPUT_DIR = "data/raw/clean_real"

def get_current_count():
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('real_') and f.endswith('.wav')]
    return len(existing)

def download_from_proven_datasets():
    """Download only from datasets that we know work"""
    
    current_count = get_current_count()
    target_total = 1000
    needed = target_total - current_count
    
    if needed <= 0:
        print(f"✅ Already have {current_count} voices (target: {target_total})")
        return
    
    print(f"🎯 Need {needed} more voices")
    
    # Use only proven working datasets
    working_datasets = [
        ("librispeech_asr", "clean", "train.360", min(300, needed)),  # Different split
        ("librispeech_asr", "clean", "train.500", min(200, needed)),  # Another split
        ("speech_commands", "v0.02", "train", min(200, needed)),      # More from this
        ("google/fleurs", "en_us", "train", min(200, needed)),        # More from this
    ]
    
    total_added = 0
    
    for dataset_name, config, split, target_count in working_datasets:
        if total_added >= needed:
            break
            
        print(f"📥 Getting {target_count} from {dataset_name} {split}...")
        added = download_small_dataset(dataset_name, config, split, target_count)
        total_added += added
    
    final_count = get_current_count()
    print(f"\n🎉 Added {total_added} new voices")
    print(f"📊 Total: {final_count} voices")

# Reuse the working download function
def download_small_dataset(dataset_name, config, split, num_samples):
    from datasets import load_dataset
    
    try:
        dataset = load_dataset(dataset_name, config, split=split, streaming=True, trust_remote_code=True)
        start_idx = get_current_count() + 1
        saved_count = 0
        
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
                
            try:
                audio_array = sample['audio']['array']
                sr = sample['audio']['sampling_rate']
                
                if sr != TARGET_SR:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)
                
                if len(audio_array) > TARGET_SAMPLES:
                    audio_array = audio_array[:TARGET_SAMPLES]
                else:
                    padding = TARGET_SAMPLES - len(audio_array)
                    audio_array = np.pad(audio_array, (0, padding))
                
                filename = f"real_{start_idx + saved_count:04d}.wav"
                output_path = os.path.join(OUTPUT_DIR, filename)
                sf.write(output_path, audio_array, TARGET_SR)
                saved_count += 1
                
                if saved_count % 50 == 0:
                    print(f"   ✅ Saved {saved_count}/{num_samples}")
                    
            except Exception as e:
                continue
        
        print(f"✅ Saved {saved_count} from {dataset_name}")
        return saved_count
        
    except Exception as e:
        print(f"❌ {dataset_name} failed: {e}")
        return 0

if __name__ == "__main__":
    download_from_proven_datasets()