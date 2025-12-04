# scripts/cleanup_ai_voices.py
import os
import librosa
import soundfile as sf
import numpy as np

def cleanup_ai_voices():
    """
    Simple cleanup for AI voices
    - Normalize all to 64600 samples (4.04s)
    - Ensure 16kHz sample rate
    - Keep existing ai_XXXX.wav names
    """
    
    # Your AI directory
    ai_dir = r"D:\sk13382\DataTraining\aasist\noise_robust_detector\data\raw\clean_ai"
    
    print(f"📁 Working on: {ai_dir}")
    
    if not os.path.exists(ai_dir):
        print(f"❌ Directory doesn't exist: {ai_dir}")
        return
    
    # Get all WAV files
    files = [f for f in os.listdir(ai_dir) if f.endswith('.wav')]
    print(f"Found {len(files)} WAV files")
    
    if not files:
        print("❌ No WAV files found")
        return
    
    # Settings
    SAMPLE_RATE = 16000
    TARGET_LENGTH = 64600  # 4.04 seconds
    
    processed = 0
    errors = []
    
    for i, filename in enumerate(files):
        filepath = os.path.join(ai_dir, filename)
        
        try:
            # Load audio
            audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
            
            # Check current length
            current_len = len(audio)
            
            if current_len == TARGET_LENGTH and sr == SAMPLE_RATE:
                # Already correct, skip
                if i < 5:  # Show first 5
                    print(f"✓ {filename}: Already correct ({current_len} samples)")
                continue
            
            # Fix length
            if current_len > TARGET_LENGTH:
                # Trim from center
                start = (current_len - TARGET_LENGTH) // 2
                audio = audio[start:start + TARGET_LENGTH]
            elif current_len < TARGET_LENGTH:
                # Pad with silence
                padding = TARGET_LENGTH - current_len
                audio = np.pad(audio, (0, padding), mode='constant')
            
            # Save back (overwrite)
            sf.write(filepath, audio, SAMPLE_RATE, subtype='PCM_16')
            processed += 1
            
            if i < 5:  # Show first 5
                print(f"✓ {filename}: Fixed ({current_len} → {len(audio)} samples)")
                
        except Exception as e:
            errors.append(f"{filename}: {e}")
    
    print(f"\n✅ Done!")
    print(f"Processed: {processed} files")
    print(f"Already correct: {len(files) - processed - len(errors)} files")
    
    if errors:
        print(f"\n❌ Errors ({len(errors)}):")
        for error in errors[:5]:
            print(f"  {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    # Quick verification
    print(f"\n🔍 Verifying first 5 files...")
    for i, filename in enumerate(files[:5]):
        filepath = os.path.join(ai_dir, filename)
        try:
            audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
            if len(audio) == TARGET_LENGTH and sr == SAMPLE_RATE:
                print(f"  ✓ {filename}: {len(audio)} samples, {sr}Hz")
            else:
                print(f"  ✗ {filename}: {len(audio)} samples, {sr}Hz (should be {TARGET_LENGTH}, {SAMPLE_RATE})")
        except Exception as e:
            print(f"  ✗ {filename}: ERROR - {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("AI VOICES CLEANUP - SIMPLE VERSION")
    print("=" * 50)
    cleanup_ai_voices()