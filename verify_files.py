import os
import librosa

print("🔍 Verifying AI audio files...")

ai_files = os.listdir('data/ai')
print(f"Found {len(ai_files)} files in data/ai/")

for filename in ai_files:
    if filename.endswith('.wav'):
        filepath = os.path.join('data/ai', filename)
        try:
            audio, sr = librosa.load(filepath, sr=16000)
            duration = len(audio) / sr
            print(f"✅ {filename}: {duration:.2f}s, {sr}Hz")
        except Exception as e:
            print(f"❌ {filename}: Error - {e}")

print("✅ Verification complete")