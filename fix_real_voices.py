import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

def enhance_real_voices():
    """Fix volume and quality issues in real voices"""
    print("=== ENHANCING REAL VOICE QUALITY ===")
    
    input_folder = Path("data_normalized/real")
    output_folder = Path("data_enhanced/real")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    files = list(input_folder.glob("*.wav"))
    print(f"Processing {len(files)} real voice files...")
    
    for file_path in files:
        # Load audio
        audio, sr = librosa.load(file_path, sr=16000)
        
        print(f"\n{file_path.name}:")
        print(f"  Before - Max: {np.max(np.abs(audio)):.3f}, Mean: {np.mean(np.abs(audio)):.4f}")
        
        # Remove excessive silence (threshold very low sounds)
        threshold = 0.005  # Very conservative threshold
        audio_clean = audio.copy()
        audio_clean[np.abs(audio_clean) < threshold] = 0
        
        # Calculate how much to amplify (target peak of 0.8)
        current_peak = np.max(np.abs(audio_clean))
        if current_peak > 0:  # Avoid division by zero
            amplification = 0.8 / current_peak
            # Don't over-amplify (max 10x)
            amplification = min(amplification, 10.0)
        else:
            amplification = 1.0
        
        audio_enhanced = audio_clean * amplification
        
        # Light noise reduction (very gentle)
        audio_final = librosa.effects.preemphasis(audio_enhanced, coef=0.95)
        
        print(f"  After  - Max: {np.max(np.abs(audio_final)):.3f}, Mean: {np.mean(np.abs(audio_final)):.4f}")
        print(f"  Amplification: {amplification:.1f}x")
        
        # Save enhanced file
        output_path = output_folder / file_path.name
        sf.write(output_path, audio_final, sr)
    
    print(f"\n✅ Enhanced real voices saved to: {output_folder}")

def compare_quality():
    """Compare original vs enhanced voices"""
    print("\n=== QUALITY COMPARISON ===")
    
    original_folder = Path("data_normalized/real")
    enhanced_folder = Path("data_enhanced/real")
    
    print("REAL VOICES - ORIGINAL vs ENHANCED:")
    
    # Convert generator to list and take first 2
    files = list(original_folder.glob("*.wav"))[:2]
    
    for file_path in files:
        # Original
        audio_orig, sr = librosa.load(file_path, sr=16000)
        
        # Enhanced
        enhanced_path = enhanced_folder / file_path.name
        audio_enh, _ = librosa.load(enhanced_path, sr=16000)
        
        print(f"\n{file_path.name}:")
        print(f"  Original: Max {np.max(np.abs(audio_orig)):.3f}, Mean {np.mean(np.abs(audio_orig)):.4f}")
        print(f"  Enhanced: Max {np.max(np.abs(audio_enh)):.3f}, Mean {np.mean(np.abs(audio_enh)):.4f}")

if __name__ == "__main__":
    enhance_real_voices()
    compare_quality()