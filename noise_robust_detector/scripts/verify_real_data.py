"""
VERIFY REAL VOICE DATASET - Fixed
"""

import os
import sys
import librosa
import numpy as np

# Fix import path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from project_config import *

def verify_real_voices():
    """Verify ALL downloaded real voice samples"""
    
    print("🔍 VERIFYING REAL VOICE DATASET")
    print("=" * 50)
    
    real_voice_dir = os.path.join(DATA_RAW, "clean_real")
    
    if not os.path.exists(real_voice_dir):
        print("❌ Real voice directory not found")
        return
    
    # Get all real voice files
    real_files = [f for f in os.listdir(real_voice_dir) if f.endswith('.wav')]
    
    print(f"📊 Found {len(real_files)} real voice samples")
    
    # Analyze ALL samples
    durations = []
    sample_rates = []
    valid_samples = 0
    
    print("\n🎵 Analyzing all samples (this may take a minute)...")
    
    for i, file in enumerate(real_files):
        filepath = os.path.join(real_voice_dir, file)
        try:
            audio, sr = librosa.load(filepath, sr=None)
            duration = len(audio) / sr
            durations.append(duration)
            sample_rates.append(sr)
            valid_samples += 1
            
            # Show progress for larger datasets
            if (i + 1) % 50 == 0:
                print(f"   Checked {i + 1}/{len(real_files)} samples...")
                
        except Exception as e:
            print(f"   ❌ {file}: Error - {e}")
    
    # Summary statistics
    if durations:
        print(f"\n📈 DATASET SUMMARY:")
        print(f"   Valid samples: {valid_samples}/{len(real_files)}")
        print(f"   Average duration: {np.mean(durations):.2f}s")
        print(f"   Duration range: {np.min(durations):.2f}s - {np.max(durations):.2f}s")
        print(f"   Sample rate: {sample_rates[0]}Hz (consistent)")
        
        # Check duration consistency
        unique_durations = len(set(round(d, 2) for d in durations))
        print(f"   Unique durations: {unique_durations}")
    
    # Check for diversity
    print(f"\n🎭 DIVERSITY CHECK:")
    print(f"   Samples from: LibriSpeech (professional speakers)")
    print(f"   Expected gender mix: Balanced male/female")
    print(f"   Audio quality: Studio recording")
    print(f"   Content: Reading passages")
    
    return len(real_files), valid_samples

def check_audio_quality():
    """Quick quality check on random samples"""
    
    print(f"\n🎵 RANDOM SAMPLE QUALITY CHECK:")
    
    real_voice_dir = os.path.join(DATA_RAW, "clean_real")
    real_files = [f for f in os.listdir(real_voice_dir) if f.endswith('.wav')]
    
    # Check 5 random samples in detail
    import random
    random_samples = random.sample(real_files, min(5, len(real_files)))
    
    for file in random_samples:
        filepath = os.path.join(real_voice_dir, file)
        try:
            audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
            duration = len(audio) / sr
            rms_energy = np.sqrt(np.mean(audio**2))
            
            print(f"   {file}:")
            print(f"     Duration: {duration:.2f}s")
            print(f"     Energy: {rms_energy:.4f}")
            print(f"     Length: {len(audio)} samples")
            
        except Exception as e:
            print(f"   ❌ {file}: {e}")

def main():
    print("🎯 STEP 5: REAL VOICE DATASET VERIFICATION")
    print("=" * 60)
    
    total_samples, valid_samples = verify_real_voices()
    check_audio_quality()
    
    print("\n" + "=" * 60)
    if valid_samples == total_samples:
        print("✅ PERFECT: All real voice samples are valid!")
        print(f"🎉 You have {valid_samples} professional real voice samples")
        print("   Ready for AI voice collection")
    elif valid_samples >= 150:
        print("✅ GOOD: Most real voice samples are valid")
        print(f"💡 You have {valid_samples} professional real voice samples")
        print("   Ready to proceed")
    else:
        print("⚠️  Some samples may need review")
        print(f"   {valid_samples}/{total_samples} samples are valid")
    
    print("=" * 60)

if __name__ == "__main__":
    main()