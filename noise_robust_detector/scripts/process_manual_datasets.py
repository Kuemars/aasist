import os
import glob
import librosa
import soundfile as sf
import numpy as np
import random

def process_manual_datasets(input_folders, num_samples_per_folder=100):
    """
    Process manually downloaded datasets and add to real voices
    """
    OUTPUT_DIR = "data/raw/clean_real"
    TARGET_SR = 16000
    TARGET_SAMPLES = 4 * TARGET_SR  # 4 seconds
    
    current_count = len([f for f in os.listdir(OUTPUT_DIR) if f.startswith('real_') and f.endswith('.wav')])
    
    total_added = 0
    
    for folder in input_folders:
        if not os.path.exists(folder):
            print(f"❌ Folder not found: {folder}")
            continue
            
        print(f"📁 Processing: {folder}")
        
        # Find all audio files
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']:
            audio_files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
        
        if not audio_files:
            print(f"   No audio files found in {folder}")
            continue
            
        print(f"   Found {len(audio_files)} audio files")
        
        # Select random samples
        selected_files = random.sample(audio_files, min(num_samples_per_folder, len(audio_files)))
        
        added_from_folder = 0
        
        for audio_path in selected_files:
            try:
                # Normalize audio
                audio, sr = librosa.load(audio_path, sr=TARGET_SR)
                
                # Trim or pad
                if len(audio) > TARGET_SAMPLES:
                    audio = audio[:TARGET_SAMPLES]
                else:
                    padding = TARGET_SAMPLES - len(audio)
                    audio = np.pad(audio, (0, padding))
                
                # Save
                filename = f"real_{current_count + total_added + 1:04d}.wav"
                output_path = os.path.join(OUTPUT_DIR, filename)
                sf.write(output_path, audio, TARGET_SR)
                
                added_from_folder += 1
                total_added += 1
                
            except Exception as e:
                print(f"❌ Error processing {os.path.basename(audio_path)}: {e}")
        
        print(f"   ✅ Added {added_from_folder} samples from {folder}")
    
    print(f"\n🎉 Total added: {total_added}")
    print(f"📊 New total: {current_count + total_added}")

if __name__ == "__main__":
    # Example usage - modify these paths
    manual_folders = [
        "/path/to/your/voxceleb/download",
        "/path/to/your/commonvoice/download", 
        "/path/to/your/other/datasets"
    ]
    
    process_manual_datasets(manual_folders, num_samples_per_folder=100)