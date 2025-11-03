import os
import librosa
import soundfile as sf

print("🔄 Renaming and converting real voice files...")

real_files = os.listdir('data/real')
print(f"Found {len(real_files)} files to process")

# Process each file
for i, filename in enumerate(real_files):
    old_path = os.path.join('data/real', filename)
    
    # New name format: real_001.wav, real_002.wav, etc.
    new_name = f"real_{i+1:03d}.wav"
    new_path = os.path.join('data/real', new_name)
    
    try:
        # If it's already a WAV file, just rename it
        if filename.endswith('.wav'):
            os.rename(old_path, new_path)
            print(f"✅ Renamed: {filename} -> {new_name}")
        
        # If it's FLAC, convert to WAV
        elif filename.endswith('.flac'):
            audio, sr = librosa.load(old_path, sr=16000)
            sf.write(new_path, audio, sr)
            os.remove(old_path)  # Remove the original FLAC
            print(f"✅ Converted: {filename} -> {new_name}")
            
        else:
            print(f"⚠️  Skipped (not WAV/FLAC): {filename}")
            
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")

print("✅ Renaming and conversion complete")