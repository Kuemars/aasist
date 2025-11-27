# scripts/download_eleven_labs.py
import os
import requests
import zipfile
import tarfile
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import project_config_extended as config

class ElevenLabsDownloader:
    def __init__(self):
        self.target_duration = config.TARGET_DURATION
        self.sample_rate = config.SAMPLE_RATE
        self.target_samples = config.AUDIO_LENGTH
        
    def download_eleven_labs_dataset(self):
        """Download Eleven Labs Respeech Dataset"""
        print("🚀 DOWNLOADING ELEVEN LABS RESPEECH DATASET")
        
        url = config.AI_DATASET_URLS["eleven_labs"]
        download_dir = config.ELEVEN_LABS_RAW_DIR
        download_dir.mkdir(parents=True, exist_ok=True)
        
        filename = "eleven_labs_respeech.zip"
        filepath = download_dir / filename
        
        try:
            # Download with progress bar
            print(f"📥 Downloading from: {url}")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc="Downloading Eleven Labs",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=8192):
                    size = f.write(data)
                    pbar.update(size)
            
            print("✅ Download completed")
            
            # Extract
            print("📦 Extracting dataset...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(download_dir)
            
            print("✅ Extraction completed")
            
            return download_dir
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None
    
    def discover_audio_files(self, base_dir):
        """Discover all audio files in the dataset"""
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(Path(base_dir).rglob(f'*{ext}'))
            # Also check uppercase extensions
            audio_files.extend(Path(base_dir).rglob(f'*{ext.upper()}'))
        
        print(f"🎵 Found {len(audio_files)} audio files in Eleven Labs dataset")
        return audio_files
    
    def normalize_audio_files(self, audio_files, output_dir, prefix="eleven_"):
        """Normalize audio files to 4s, 16kHz WAV"""
        output_dir.mkdir(parents=True, exist_ok=True)
        processed_files = []
        
        for i, audio_path in enumerate(tqdm(audio_files, desc="Normalizing Eleven Labs audio")):
            try:
                # Load audio
                audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
                
                # Process audio to target duration
                if len(audio) > self.target_samples:
                    # Trim to target duration
                    audio = audio[:self.target_samples]
                else:
                    # Pad with silence
                    padding = self.target_samples - len(audio)
                    audio = np.pad(audio, (0, padding), mode='constant')
                
                # Normalize amplitude
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio)) * 0.9
                
                # Save as WAV
                output_filename = f"{prefix}{i+1:04d}.wav"
                output_path = output_dir / output_filename
                
                sf.write(output_path, audio, sr, subtype='PCM_16')
                processed_files.append(output_path)
                
            except Exception as e:
                print(f"❌ Error processing {audio_path}: {e}")
                continue
        
        return processed_files
    
    def process_eleven_labs_dataset(self):
        """Main processing function for Eleven Labs dataset"""
        # Download dataset
        dataset_dir = self.download_eleven_labs_dataset()
        if not dataset_dir:
            return []
        
        # Discover audio files
        audio_files = self.discover_audio_files(dataset_dir)
        
        if not audio_files:
            print("❌ No audio files found in Eleven Labs dataset")
            return []
        
        # Normalize audio files
        output_dir = config.AI_RAW_DIR / "eleven_labs_processed"
        processed_files = self.normalize_audio_files(audio_files, output_dir, "eleven_")
        
        print(f"✅ Successfully processed {len(processed_files)} Eleven Labs AI voices")
        return processed_files

def main():
    downloader = ElevenLabsDownloader()
    return downloader.process_eleven_labs_dataset()

if __name__ == "__main__":
    main()