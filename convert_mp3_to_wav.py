import os
import librosa
import soundfile as sf

# Convert all MP3 files in data/ai/ to WAV
for filename in os.listdir('data/ai'):
    if filename.endswith('.mp3'):
        mp3_path = os.path.join('data/ai', filename)
        wav_filename = filename.replace('.mp3', '.wav')
        wav_path = os.path.join('data/ai', wav_filename)
        
        # Load MP3 and save as WAV using librosa
        audio, sr = librosa.load(mp3_path, sr=16000)
        sf.write(wav_path, audio, sr)
        print(f"Converted: {filename} -> {wav_filename}")

print("✅ All MP3 files converted to WAV")