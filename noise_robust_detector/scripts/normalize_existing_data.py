"""
SIMPLE NORMALIZATION - Fix your 15 AI voices
"""

import os
import sys
import librosa
import soundfile as sf
import numpy as np

# Fix import path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from project_config import *

def normalize_your_ai_voices():
    """Normalize your 15 existing AI voices to match real voice length"""
    
    print("🎯 NORMALIZING YOUR 15 AI VOICES")
    print("=" * 50)
    
    # Your AI voices location
    source_dir = 'data/ai'  # From /d/seniorThesis/AasistModel/aasist/data/ai
    target_dir = os.path.join(DATA_RAW, "clean_ai")
    
    os.makedirs(target_dir, exist_ok=True)
    
    # Check if source exists
    if not os.path.exists(source_dir):
        print(f"❌ AI voice directory not found: {source_dir}")
        return 0
    
    # Get your 15 AI files
    ai_files = [f for f in os.listdir(source_dir) if f.endswith('.wav')]
    
    print(f"Found {len(ai_files)} AI voice files")
    print("Files to process:")
    for file in ai_files:
        print(f"  - {file}")
    
    processed_count = 0
    
    for file in ai_files:
        input_path = os.path.join(source_dir, file)
        output_path = os.path.join(target_dir, file)  # Keep same filename
        
        try:
            # Load audio
            audio, sr = librosa.load(input_path, sr=SAMPLE_RATE)
            original_duration = len(audio) / sr
            
            print(f"\n🔧 Processing: {file}")
            print(f"   Original: {original_duration:.2f}s, {len(audio)} samples")
            
            # Normalize to exactly 64600 samples (4.04s at 16kHz)
            if len(audio) > AUDIO_LENGTH:
                # Trim from center to keep important content
                start = (len(audio) - AUDIO_LENGTH) // 2
                audio = audio[start:start + AUDIO_LENGTH]
                print(f"   Trimmed: {len(audio)} samples")
            else:
                # Pad with silence
                padding = AUDIO_LENGTH - len(audio)
                audio = np.pad(audio, (0, padding))
                print(f"   Padded: {len(audio)} samples")
            
            # Save normalized file
            sf.write(output_path, audio, sr)
            processed_count += 1
            
            # Verify the result
            verify_audio, sr = librosa.load(output_path, sr=None)
            final_duration = len(verify_audio) / sr
            print(f"   ✅ Final: {final_duration:.2f}s, {len(verify_audio)} samples")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    return processed_count

def main():
    print("🎯 STEP 7: NORMALIZE YOUR AI VOICES")
    print("=" * 60)
    print("Making your 15 AI voices match the 4.04s length of real voices")
    print("=" * 60)
    
    processed = normalize_your_ai_voices()
    
    print("\n" + "=" * 60)
    if processed == 15:
        print("🎉 PERFECT: All 15 AI voices normalized!")
        print("💡 Now you have:")
        print("   - 200 real voices (4.04s each)")
        print("   - 15 AI voices (4.04s each)") 
        print("   - Ready for training or minimal expansion")
    elif processed > 0:
        print(f"✅ GOOD: {processed}/15 AI voices normalized")
        print("   You can proceed with this dataset")
    else:
        print("❌ No files were processed")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
