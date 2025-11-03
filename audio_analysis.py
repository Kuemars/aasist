import librosa
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_audio_characteristics():
    """Compare spectral characteristics of AI vs Real voices"""
    print("=== AUDIO CHARACTERISTICS ANALYSIS ===")
    
    data_folder = Path("data_normalized")
    
    for voice_type in ['ai', 'real']:
        folder = data_folder / voice_type
        files = list(folder.glob("*.wav"))
        
        print(f"\n--- {voice_type.upper()} VOICE ANALYSIS ---")
        
        # Analyze first 3 files in detail
        for i, file_path in enumerate(files[:3]):
            audio, sr = librosa.load(file_path, sr=16000)
            
            # Calculate various audio features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            rms_energy = librosa.feature.rms(y=audio)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            
            print(f"\n{file_path.name}:")
            print(f"  Spectral centroid: {np.mean(spectral_centroid):.1f} Hz")
            print(f"  RMS energy: {np.mean(rms_energy):.6f}")
            print(f"  Zero-crossing rate: {np.mean(zero_crossing_rate):.4f}")
            
            # Check for silence/padding
            silence_threshold = 0.001
            silent_samples = np.sum(np.abs(audio) < silence_threshold)
            silence_percentage = (silent_samples / len(audio)) * 100
            print(f"  Silence percentage: {silence_percentage:.1f}%")

def check_audio_quality():
    """Check if real voices have quality issues"""
    print("\n=== AUDIO QUALITY CHECK ===")
    
    data_folder = Path("data_normalized")
    
    for voice_type in ['real']:  # Focus on real voices
        folder = data_folder / voice_type
        
        print(f"\nChecking {voice_type} voice quality:")
        
        for file_path in folder.glob("*.wav"):
            audio, sr = librosa.load(file_path, sr=16000)
            
            # Check for common issues
            max_amplitude = np.max(np.abs(audio))
            mean_amplitude = np.mean(np.abs(audio))
            
            # Quality indicators
            if max_amplitude < 0.1:
                print(f"  ⚠️  {file_path.name}: LOW VOLUME (max: {max_amplitude:.3f})")
            elif mean_amplitude < 0.01:
                print(f"  ⚠️  {file_path.name}: VERY QUIET (mean: {mean_amplitude:.3f})")
            
            # Check for excessive noise
            if np.std(audio) < 0.005:
                print(f"  ⚠️  {file_path.name}: MAYBE OVER-COMPRESSED (std: {np.std(audio):.4f})")

if __name__ == "__main__":
    analyze_audio_characteristics()
    check_audio_quality()