"""
LIBRISPEECH DOWNLOAD AUTOMATION - Space Aware
"""

import os
import sys
import requests
import tarfile
import librosa
import soundfile as sf
import numpy as np

# Fix import path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from project_config import *

def check_disk_space():
    """Check if we have enough disk space"""
    print("💾 Checking disk space...")
    
    # Check available space in current directory
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (2**30)  # Convert to GB
        
        print(f"   Available space: {free_gb} GB")
        
        # LibriSpeech requires ~6.3GB download + ~6GB extracted + processing
        if free_gb < 15:
            print("❌ WARNING: Less than 15GB free space")
            print("   Consider downloading to a different drive")
            return False
        else:
            print("✅ Sufficient disk space available")
            return True
            
    except:
        print("⚠️  Could not check disk space")
        return True  # Continue anyway

def download_librispeech():
    """Download and process LibriSpeech dataset"""
    
    print("🎯 DOWNLOADING LIBRISPEECH DATASET")
    print("=" * 50)
    
    # Check disk space first
    if not check_disk_space():
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            return False
    
    # Download URL for train-clean-100 (100 hours)
    url = "http://www.openslr.org/resources/12/train-clean-100.tar.gz"
    filename = "train-clean-100.tar.gz"
    
    print(f"\n📥 Downloading: {url}")
    print("   File size: ~6.3GB")
    print("   Total space needed: ~15GB (download + extraction + processing)")
    print("   This may take 10-30 minutes...")
    
    # Download the file
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f:
            downloaded = 0
            last_percent = -1
            
            for data in response.iter_content(chunk_size=8192):
                downloaded += len(data)
                f.write(data)
                
                # Progress indicator - only print every 10%
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    current_percent = int(percent // 10) * 10
                    
                    if current_percent != last_percent:
                        print(f"   Progress: {current_percent}%")
                        last_percent = current_percent
        
        print("✅ Download complete!")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False
    
    # Extract the archive
    print("📦 Extracting archive (this may take a few minutes)...")
    try:
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall()
        print("✅ Extraction complete!")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False
    
    # Process the files
    print("🔄 Processing audio files...")
    process_librispeech_files()
    
    return True

def process_librispeech_files():
    """Convert LibriSpeech FLAC files to WAV format - process in batches to save memory"""
    
    output_dir = os.path.join(DATA_RAW, "clean_real")
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    target_count = 200  # Get 200 diverse samples
    
    print("   Converting FLAC to WAV...")
    
    # Process in smaller batches to manage memory
    batch_size = 50
    current_batch = 0
    
    # Walk through extracted directory
    for root, dirs, files in os.walk("LibriSpeech/train-clean-100"):
        for file in files:
            if file.endswith('.flac') and count < target_count:
                input_path = os.path.join(root, file)
                output_filename = f"real_librispeech_{count:04d}.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                try:
                    # Convert FLAC to WAV
                    audio, sr = librosa.load(input_path, sr=SAMPLE_RATE)
                    
                    # Ensure correct length
                    if len(audio) > AUDIO_LENGTH:
                        audio = audio[:AUDIO_LENGTH]
                    else:
                        # Pad if too short
                        padding = AUDIO_LENGTH - len(audio)
                        audio = np.pad(audio, (0, padding))
                    
                    # Save as WAV
                    sf.write(output_path, audio, sr)
                    count += 1
                    
                    # Progress - only print every 25 samples
                    if count % 25 == 0:
                        print(f"   Processed {count}/{target_count} samples...")
                        
                except Exception as e:
                    continue
    
    print(f"✅ Successfully processed {count} LibriSpeech samples!")
    print(f"📍 Saved to: {output_dir}")

def cleanup_download_files():
    """Clean up temporary download files to free space"""
    print("\n🧹 Cleaning up temporary files to free space...")
    
    files_to_remove = [
        "train-clean-100.tar.gz",  # ~6.3GB
        "LibriSpeech"              # ~6GB extracted
    ]
    
    space_freed = 0
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                if os.path.isdir(file):
                    import shutil
                    shutil.rmtree(file)
                    # Estimate space freed
                    space_freed += 6  # GB estimate
                else:
                    os.remove(file)
                    space_freed += 6.3  # GB estimate
                print(f"✅ Removed: {file}")
            except Exception as e:
                print(f"⚠️  Could not remove {file}: {e}")
    
    print(f"💾 Freed approximately {space_freed} GB of space")

def main():
    print("🎯 LARGE-SCALE REAL VOICE COLLECTION")
    print("=" * 60)
    print("⚠️  SPACE REQUIREMENTS:")
    print("   - Download: 6.3GB")
    print("   - Extraction: 6GB") 
    print("   - Processing: Additional space")
    print("   - Final dataset: ~500MB (200 processed samples)")
    print("=" * 60)
    
    # Download and process LibriSpeech
    success = download_librispeech()
    
    if success:
        cleanup_download_files()
        print("\n🎉 LIBRISPEECH COLLECTION COMPLETE!")
        print("💡 You now have 200+ professional real voice samples")
        print("   Final dataset size: ~500MB")
        print("   Next: Add Common Voice for even more diversity")
    else:
        print("\n❌ Download failed.")

if __name__ == "__main__":
    main()