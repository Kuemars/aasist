import os
import sys
import zipfile
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import random
from tqdm import tqdm

# Configuration
TARGET_SR = 16000
TARGET_DURATION = 4.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)
OUTPUT_DIR = "data/raw/clean_real"

def get_current_count():
    """Get current number of real voice files"""
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('real_') and f.endswith('.wav')]
    return len(existing)

def get_audio_files_from_zip(zip_path):
    """Get all audio file paths from the zip"""
    print("🔍 Finding audio files in zip...")
    
    audio_files = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get all MP3 files (they're all in cv-invalid/cv-invalid/ folder)
            for file_name in zip_ref.namelist():
                if file_name.endswith('.mp3'):
                    audio_files.append(file_name)
    
    except Exception as e:
        print(f"❌ Error reading zip: {e}")
        return []
    
    print(f"📁 Found {len(audio_files)} audio files")
    return audio_files

def process_samples(zip_path, audio_files, num_samples=1000):
    """Process and save audio samples"""
    print(f"🎯 Processing {num_samples} random samples...")
    
    current_count = get_current_count()
    start_count = current_count + 1
    added_count = 0
    
    # Select random samples
    selected_files = random.sample(audio_files, min(num_samples, len(audio_files)))
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for i, audio_path_in_zip in enumerate(tqdm(selected_files, desc="Processing")):
            try:
                # Extract and process the audio
                with zip_ref.open(audio_path_in_zip) as audio_file:
                    # Save MP3 temporarily
                    temp_mp3 = f"temp_{i}.mp3"
                    with open(temp_mp3, 'wb') as f:
                        f.write(audio_file.read())
                    
                    # Load and convert to WAV
                    audio, sr = librosa.load(temp_mp3, sr=TARGET_SR)
                    
                    # Clean up temp file
                    os.remove(temp_mp3)
                    
                    # Skip if too short (less than 2 seconds)
                    if len(audio) < TARGET_SR * 2:
                        continue
                    
                    # Normalize to 4 seconds
                    if len(audio) > TARGET_SAMPLES:
                        # Trim from center
                        start = (len(audio) - TARGET_SAMPLES) // 2
                        audio = audio[start:start + TARGET_SAMPLES]
                    else:
                        # Pad with silence
                        padding = TARGET_SAMPLES - len(audio)
                        audio = np.pad(audio, (0, padding))
                    
                    # Save the file
                    filename = f"real_{start_count + added_count:04d}.wav"
                    output_path = os.path.join(OUTPUT_DIR, filename)
                    sf.write(output_path, audio, TARGET_SR)
                    
                    added_count += 1
                    
                    # Progress update every 100 files
                    if added_count % 100 == 0:
                        print(f"   ✅ Saved {added_count}/{num_samples}")
                        
            except Exception as e:
                # Skip files that can't be processed
                continue
    
    return added_count

def main():
    print("🚀 PROCESSING 1000 COMMON VOICE INVALID SAMPLES")
    print("=" * 60)
    
    zip_path = "D:/sk13382/archive.zip"
    
    if not os.path.exists(zip_path):
        print(f"❌ Zip file not found: {zip_path}")
        return
    
    print(f"✅ Found zip file: {zip_path}")
    
    # Step 1: Get all audio files
    audio_files = get_audio_files_from_zip(zip_path)
    if not audio_files:
        print("❌ No audio files found")
        return
    
    # Step 2: Process samples
    added_count = process_samples(zip_path, audio_files, 1000)
    
    # Final results
    final_count = get_current_count()
    print(f"\n🎉 PROCESSING COMPLETE!")
    print(f"📊 Added {added_count} new real voices from Common Voice invalid")
    print(f"📈 Total real voices: {final_count}")

if __name__ == "__main__":
    main()