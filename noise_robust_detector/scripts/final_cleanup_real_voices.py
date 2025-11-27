import os
import librosa
import soundfile as sf
import numpy as np
import re

def final_cleanup_real_voices():
    """
    FINAL CLEANUP: 
    - Convert all files to WAV format
    - Rename everything to real_XXXX.wav with consecutive numbering
    - Normalize all to 4 seconds, 16kHz
    - Remove any corrupted files
    """
    
    real_voices_dir = "data/raw/clean_real"
    
    if not os.path.exists(real_voices_dir):
        print(f"❌ Directory not found: {real_voices_dir}")
        return
    
    print("🧹 FINAL CLEANUP OF REAL VOICES DATASET")
    print("=" * 50)
    
    # Get ALL files in the directory
    all_items = os.listdir(real_voices_dir)
    print(f"📁 Found {len(all_items)} items in directory")
    
    # Separate files from anything else
    files_to_process = []
    other_items = []
    
    for item in all_items:
        item_path = os.path.join(real_voices_dir, item)
        if os.path.isfile(item_path):
            files_to_process.append(item)
        else:
            other_items.append(item)
    
    if other_items:
        print(f"⚠️  Found {len(other_items)} non-file items (directories, etc.)")
        for item in other_items:
            print(f"   🗑️  Removing: {item}")
            item_path = os.path.join(real_voices_dir, item)
            # Try to remove (might be directories)
            try:
                if os.path.isdir(item_path):
                    import shutil
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except:
                print(f"   ❌ Could not remove: {item}")
    
    print(f"📄 Processing {len(files_to_process)} files...")
    
    # Step 1: Convert all files to WAV and normalize
    converted_files = []
    valid_audio_files = []
    
    for filename in files_to_process:
        filepath = os.path.join(real_voices_dir, filename)
        
        try:
            # Try to load as audio file regardless of extension
            audio, sr = librosa.load(filepath, sr=16000)
            
            # Normalize to 4 seconds
            target_samples = 4 * 16000  # 64000 samples
            
            if len(audio) > target_samples:
                # Trim from center
                start = (len(audio) - target_samples) // 2
                audio = audio[start:start + target_samples]
            else:
                # Pad with silence
                padding = target_samples - len(audio)
                audio = np.pad(audio, (0, padding))
            
            # Create temporary WAV file
            temp_wav = f"temp_{len(converted_files)}.wav"
            temp_path = os.path.join(real_voices_dir, temp_wav)
            sf.write(temp_path, audio, 16000)
            
            # Remove original file
            os.remove(filepath)
            
            converted_files.append(temp_wav)
            valid_audio_files.append((temp_wav, audio))
            
            print(f"✅ Converted: {filename} -> {temp_wav}")
            
        except Exception as e:
            print(f"❌ Cannot process {filename}: {e}")
            # Remove corrupted file
            try:
                os.remove(filepath)
                print(f"🗑️  Removed corrupted: {filename}")
            except:
                print(f"❌ Cannot remove: {filename}")
    
    print(f"📊 Converted {len(converted_files)} files to WAV format")
    
    # Step 2: Rename all files to consecutive numbering
    print(f"\n🔢 Renaming all files to consecutive numbering...")
    
    for i, (temp_filename, audio) in enumerate(valid_audio_files):
        old_path = os.path.join(real_voices_dir, temp_filename)
        new_filename = f"real_{i+1:04d}.wav"
        new_path = os.path.join(real_voices_dir, new_filename)
        
        try:
            os.rename(old_path, new_path)
            print(f"📝 {temp_filename} -> {new_filename}")
        except Exception as e:
            print(f"❌ Error renaming {temp_filename}: {e}")
    
    # Final verification
    final_files = [f for f in os.listdir(real_voices_dir) if f.startswith('real_') and f.endswith('.wav')]
    valid_count = 0
    
    for filename in final_files:
        filepath = os.path.join(real_voices_dir, filename)
        try:
            audio, sr = librosa.load(filepath, sr=16000)
            if len(audio) == 64000 and sr == 16000:
                valid_count += 1
        except:
            # Remove invalid files
            os.remove(filepath)
            print(f"🗑️  Removed invalid during verification: {filename}")
    
    print(f"\n🎉 FINAL CLEANUP COMPLETE!")
    print(f"📊 Final valid real voices: {valid_count}")
    print(f"💾 All files are now: real_XXXX.wav, 4 seconds, 16kHz")
    
    return valid_count

if __name__ == "__main__":
    final_count = final_cleanup_real_voices()
    print(f"\n✅ Dataset is now perfectly clean and consistent!")
    print(f"🎯 Ready for training with {final_count} real voices")