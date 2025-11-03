import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

print("🔍 Analyzing the detection issue...")

def analyze_audio_features(audio_path, label):
    """Analyze audio features to understand the problem"""
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract basic features
        duration = len(audio) / sr
        rms_energy = np.sqrt(np.mean(audio**2))
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
        
        print(f"📄 {os.path.basename(audio_path)} ({label}):")
        print(f"   Duration: {duration:.2f}s")
        print(f"   RMS Energy: {rms_energy:.6f}")
        print(f"   Spectral Centroid: {spectral_centroid:.1f} Hz")
        return duration, rms_energy, spectral_centroid
        
    except Exception as e:
        print(f"❌ Error analyzing {audio_path}: {e}")
        return 0, 0, 0

print("\nAnalyzing AI voices:")
ai_features = []
for i in range(1, 4):  # Analyze first 3 AI files
    filepath = f"data/ai/ai_{i:03d}.wav"
    features = analyze_audio_features(filepath, "AI")
    ai_features.append(features)

print("\nAnalyzing Real voices:")
real_features = []
for i in range(1, 4):  # Analyze first 3 real files
    filepath = f"data/real/real_{i:03d}.wav"
    features = analyze_audio_features(filepath, "REAL")
    real_features.append(features)

print("\n💡 The issue: All predictions show ~0.529 confidence")
print("   This suggests the model is outputting nearly identical scores")
print("   Possible causes:")
print("   1. Model needs proper pre-training weights")
print("   2. Audio preprocessing mismatch")
print("   3. Model is outputting random predictions")