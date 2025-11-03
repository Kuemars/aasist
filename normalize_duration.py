import librosa
import librosa.util
import soundfile as sf
import numpy as np
from pathlib import Path

def normalize_audio_duration():
    """Normalize all audio files to 8 seconds duration"""
    target_duration = 8.0  # seconds
    target_sr = 16000
    target_samples = int(target_duration * target_sr)  # 128000 samples
    
    print("=== NORMALIZING AUDIO DURATIONS ===")
    print(f"Target: {target_duration}s ({target_samples} samples)")
    
    for voice_type in ['ai', 'real']:
        input_folder = Path(f"data/{voice_type}")
        output_folder = Path(f"data_normalized/{voice_type}")
        output_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"\n--- Processing {voice_type} voices ---")
        
        file_count = 0
        for file_path in input_folder.glob("*.wav"):
            # Load audio
            audio, sr = librosa.load(file_path, sr=target_sr)
            
            # Normalize duration
            if len(audio) > target_samples:
                # Take middle section for stability
                start = (len(audio) - target_samples) // 2
                audio = audio[start:start + target_samples]
            else:
                # Pad with silence
                padding = target_samples - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
            
            # Save normalized file
            output_path = output_folder / file_path.name
            sf.write(output_path, audio, target_sr)
            print(f"  {file_path.name}: {len(audio)/target_sr:.2f}s")
            file_count += 1
        
        print(f"Processed {file_count} files")
    
    print(f"\n✅ All files normalized to {target_duration} seconds")
    print("📁 Output saved to: data_normalized/")

if __name__ == "__main__":
    normalize_audio_duration()