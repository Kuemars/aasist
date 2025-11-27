import os
import sys
import zipfile
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import random
from tqdm import tqdm
import shutil

# Configuration
TARGET_SR = 16000
TARGET_DURATION = 4.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)
OUTPUT_DIR = "data/raw/clean_real"

def get_current_count():
    """Get current number of real voice files"""
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('real_') and f.endswith('.wav')]
    return len(existing)

def load_metadata_from_zip(zip_path):
    """Load metadata files directly from zip without full extraction"""
    print("📊 Loading metadata from zip...")
    
    metadata_files = {}
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Look for metadata files
            for file_name in zip_ref.namelist():
                if file_name.endswith('.tsv'):
                    print(f"   Found: {file_name}")
                    with zip_ref.open(file_name) as f:
                        try:
                            # Read the TSV file
                            content = f.read().decode('utf-8')
                            df = pd.read_csv(pd.io.common.StringIO(content), sep='\t')
                            metadata_files[file_name] = df
                            print(f"     Loaded {len(df)} entries")
                        except Exception as e:
                            print(f"     Error reading {file_name}: {e}")
    
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return {}
    
    return metadata_files

def get_validated_samples_from_metadata(metadata_files):
    """Extract validated high-quality samples from metadata"""
    print("\n🎯 Finding validated samples...")
    
    validated_samples = []
    
    for file_name, df in metadata_files.items():
        print(f"📋 Processing {os.path.basename(file_name)}...")
        
        # Common Voice columns
        if 'path' in df.columns:
            # Filter for high-quality samples
            if 'up_votes' in df.columns and 'down_votes' in df.columns:
                # Get samples with positive votes and no negative votes
                quality_samples = df[(df['up_votes'] > 0) & (df['down_votes'] == 0)]
                print(f"   High-quality samples: {len(quality_samples)}")
            else:
                quality_samples = df
                print(f"   All samples: {len(quality_samples)}")
            
            # Add to our list
            for _, row in quality_samples.iterrows():
                # Construct the path to the audio file in zip
                audio_path_in_zip = os.path.join(os.path.dirname(file_name), 'clips', row['path'])
                validated_samples.append(audio_path_in_zip)
    
    print(f"📊 Total validated samples found: {len(validated_samples)}")
    return validated_samples

def process_and_save_samples(zip_path, validated_samples, num_samples=1000):
    """Process and save the audio samples"""
    print(f"\n🎯 Processing {num_samples} samples...")
    
    current_count = get_current_count()
    start_count = current_count + 1
    added_count = 0
    
    # Select samples to process
    samples_to_process = validated_samples[:num_samples]
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for i, audio_path_in_zip in enumerate(tqdm(samples_to_process, desc="Processing")):
            try:
                # Check if file exists in zip
                if audio_path_in_zip not in zip_ref.namelist():
                    # Try alternative path (sometimes clips are in root)
                    alt_path = audio_path_in_zip.replace('clips/', '')
                    if alt_path in zip_ref.namelist():
                        audio_path_in_zip = alt_path
                    else:
                        continue
                
                # Extract and process the audio
                with zip_ref.open(audio_path_in_zip) as audio_file:
                    # Get file extension to determine format
                    file_ext = os.path.splitext(audio_path_in_zip)[1].lower()
                    
                    if file_ext == '.mp3':
                        # For MP3, save temporarily and load
                        temp_mp3 = f"temp_{i}.mp3"
                        with open(temp_mp3, 'wb') as f:
                            f.write(audio_file.read())
                        
                        audio, sr = librosa.load(temp_mp3, sr=TARGET_SR)
                        os.remove(temp_mp3)
                    else:
                        # For other formats, try direct loading
                        audio, sr = librosa.load(audio_file, sr=TARGET_SR)
                    
                    # Skip if too short
                    if len(audio) < TARGET_SR * 2:  # Less than 2 seconds
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
    print("🚀 PROCESSING 1000 COMMON VOICE SAMPLES")
    print("=" * 60)
    
    # Use Windows path format
    zip_path = "D:/sk13382/archive.zip"
    
    if not os.path.exists(zip_path):
        print(f"❌ Zip file not found: {zip_path}")
        print("💡 Please check the file path and try again")
        return
    
    print(f"✅ Found zip file: {zip_path}")
    
    # Step 1: Load metadata
    metadata_files = load_metadata_from_zip(zip_path)
    if not metadata_files:
        print("❌ No metadata files found")
        return
    
    # Step 2: Get validated samples
    validated_samples = get_validated_samples_from_metadata(metadata_files)
    if not validated_samples:
        print("❌ No validated samples found")
        return
    
    # Step 3: Process and save samples
    added_count = process_and_save_samples(zip_path, validated_samples, 1000)
    
    # Final results
    final_count = get_current_count()
    print(f"\n🎉 PROCESSING COMPLETE!")
    print(f"📊 Added {added_count} new high-quality real voices")
    print(f"📈 Total real voices: {final_count}")

if __name__ == "__main__":
    main()