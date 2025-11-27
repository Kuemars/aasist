# scripts/download_asvspoof.py
import os
import requests
import zipfile
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import project_config_extended as config

class ASVSpoofDownloader:
    def __init__(self):
        self.target_duration = config.TARGET_DURATION
        self.sample_rate = config.SAMPLE_RATE
        self.target_samples = config.AUDIO_LENGTH
    
    def download_asvspoof(self):
        """Download ASVspoof 2019 dataset"""
        print("🚀 DOWNLOADING ASVSPOOF 2019 DATASET")
        
        url = config.AI_DATASET_URLS["asvspoof"]
        download_dir = config.ASVSPOOF_RAW_DIR
        download_dir.mkdir(parents=True, exist_ok=True)
        
        filename = "ASVspoof2019_LA.zip"
        filepath = download_dir / filename
        
        try:
            # Download
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc="Downloading ASVspoof",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=8192):
                    size = f.write(data)
                    pbar.update(size)
            
            # Extract
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(download_dir)
            
            return download_dir
            
        except Exception as e:
            print(f"❌ ASVspoof download failed: {e}")
            return None
    
    def extract_fake_audio_from_asvspoof(self, dataset_dir):
        """Extract only fake audio files from ASVspoof"""
        fake_files = []
        
        # ASVspoof 2019 LA dataset structure
        possible_paths = [
            dataset_dir / "LA" / "ASVspoof2019_LA_train" / "flac",
            dataset_dir / "ASVspoof2019_LA_train" / "flac",
            dataset_dir / "train" / "flac",
        ]
        
        for train_dir in possible_paths:
            if train_dir.exists():
                print(f"🔍 Searching in: {train_dir}")
                # In ASVspoof, look for fake files (they have specific attack types)
                for audio_file in train_dir.rglob("*.flac"):
                    fake_files.append(audio_file)
        
        print(f"🎭 Found {len(fake_files)} audio files in ASVspoof")
        return fake_files
    
    def process_asvspoof_dataset(self):
        """Process ASVspoof fake audio files"""
        dataset_dir = self.download_asvspoof()
        if not dataset_dir:
            return []
        
        fake_files = self.extract_fake_audio_from_asvspoof(dataset_dir)
        
        if not fake_files:
            print("❌ No audio files found in ASVspoof")
            return []
        
        # Normalize files
        output_dir = config.AI_RAW_DIR / "asvspoof_processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed_files = []
        for i, audio_path in enumerate(tqdm(fake_files, desc="Processing ASVspoof")):
            try:
                audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
                
                if len(audio) > self.target_samples:
                    audio = audio[:self.target_samples]
                else:
                    padding = self.target_samples - len(audio)
                    audio = np.pad(audio, (0, padding), mode='constant')
                
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio)) * 0.9
                
                output_filename = f"asvspoof_{i+1:04d}.wav"
                output_path = output_dir / output_filename
                
                sf.write(output_path, audio, sr, subtype='PCM_16')
                processed_files.append(output_path)
                
            except Exception as e:
                print(f"❌ Error processing {audio_path}: {e}")
                continue
        
        print(f"✅ Processed {len(processed_files)} ASVspoof voices")
        return processed_files

def main():
    downloader = ASVSpoofDownloader()
    return downloader.process_asvspoof_dataset()

if __name__ == "__main__":
    main()