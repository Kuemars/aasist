import os

# Remove all MP3 files from data/ai directory
for filename in os.listdir('data/ai'):
    if filename.endswith('.mp3'):
        mp3_path = os.path.join('data/ai', filename)
        os.remove(mp3_path)
        print(f"Removed: {filename}")

print("✅ Cleanup complete - only WAV files remain")