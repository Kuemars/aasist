import os
import librosa

print("🔍 Verifying REAL audio files...")

real_files = os.listdir('data/real')
print(f"Found {len(real_files)} files in data/real/")

for filename in real_files:
    if filename.endswith('.wav'):
        filepath = os.path.join('data/real', filename)
        try:
            audio, sr = librosa.load(filepath, sr=16000)
            duration = len(audio) / sr
            print(f"✅ {filename}: {duration:.2f}s, {sr}Hz")
        except Exception as e:
            print(f"❌ {filename}: Error - {e}")

print("✅ Real files verification complete")