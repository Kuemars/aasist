import os
import glob
import librosa
import soundfile as sf
import numpy as np
import shutil

def verify_and_clean_real_voices():
    """
    Verify all files in clean_real directory:
    - Handle directories and permission issues
    - Check if they are valid WAV files
    - Normalize to 4 seconds, 16kHz if needed
    - Rename to real_XXXX.wav pattern
    - Remove invalid files and directories
    """
    
    real_voices_dir = "data/raw/clean_real"
    
    if not os.path.exists(real_voices_dir):
        print(f"❌ Directory not found: {real_voices_dir}")
        return
    
    # Get all items (files + directories)
    all_items = os.listdir(real_voices_dir)
    print(f"📁 Found {len(all_items)} items in {real_voices_dir}")
    
    # Separate files from directories
    files = []
    directories = []
    
    for item in all_items:
        item_path = os.path.join(real_voices_dir, item)
        if os.path.isfile(item_path):
            files.append(item)
        elif os.path.isdir(item_path):
            directories.append(item)
    
    print(f"📊 Breakdown:")
    print(f"   📄 Files: {len(files)}")
    print(f"   📁 Directories: {len(directories)}")
    
    # Handle directories first
    if directories:
        print(f"\n🔄 Processing {len(directories)} directories...")
        for dir_name in directories:
            dir_path = os.path.join(real_voices_dir, dir_name)
            print(f"   Processing directory: {dir_name}")
            
            try:
                # Check if directory is empty
                dir_contents = os.listdir(dir_path)
                if not dir_contents:
                    print(f"     🗑️  Removing empty directory: {dir_name}")
                    os.rmdir(dir_path)
                else:
                    print(f"     📂 Directory contains {len(dir_contents)} items")
                    print(f"     💡 Consider manually checking: {dir_path}")
                    
            except PermissionError:
                print(f"     ❌ Permission denied for directory: {dir_name}")
            except Exception as e:
                print(f"     ❌ Error with directory {dir_name}: {e}")
    
    # Now process files
    wav_files = [f for f in files if f.lower().endswith('.wav')]
    other_files = [f for f in files if not f.lower().endswith('.wav')]
    correctly_named = [f for f in wav_files if f.startswith('real_') and f.endswith('.wav')]
    
    print(f"\n📊 File breakdown:")
    print(f"   ✅ Correctly named WAV: {len(correctly_named)}")
    print(f"   ⚠️  Other WAV files: {len(wav_files) - len(correctly_named)}")
    print(f"   ❌ Non-WAV files: {len(other_files)}")
    
    # Process non-WAV files
    converted_count = 0
    for filename in other_files:
        filepath = os.path.join(real_voices_dir, filename)
        try:
            # Skip if it's actually a directory (shouldn't happen but just in case)
            if os.path.isdir(filepath):
                continue
                
            # Try to load as audio file
            audio, sr = librosa.load(filepath, sr=16000)
            
            # Normalize to 4 seconds
            target_samples = 4 * 16000
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            else:
                padding = target_samples - len(audio)
                audio = np.pad(audio, (0, padding))
            
            # Save as WAV
            new_filename = f"temp_converted_{converted_count}.wav"
            new_filepath = os.path.join(real_voices_dir, new_filename)
            sf.write(new_filepath, audio, 16000)
            
            # Remove original
            os.remove(filepath)
            print(f"✅ Converted: {filename} -> {new_filename}")
            converted_count += 1
            
        except Exception as e:
            print(f"❌ Cannot convert {filename}: {e}")
            try:
                # Try to remove invalid file
                os.remove(filepath)
                print(f"🗑️  Deleted invalid file: {filename}")
            except PermissionError:
                print(f"🔒 Permission denied, cannot delete: {filename}")
            except Exception as e2:
                print(f"❌ Cannot delete {filename}: {e2}")
    
    # Process incorrectly named WAV files
    wav_files = [f for f in os.listdir(real_voices_dir) if f.lower().endswith('.wav') and os.path.isfile(os.path.join(real_voices_dir, f))]
    correctly_named = [f for f in wav_files if f.startswith('real_') and f.endswith('.wav')]
    incorrectly_named = [f for f in wav_files if not (f.startswith('real_') and f.endswith('.wav'))]
    
    print(f"\n🔄 Renaming {len(incorrectly_named)} incorrectly named files...")
    
    # Find next number for renaming
    if correctly_named:
        numbers = []
        for f in correctly_named:
            try:
                num_part = f.split('_')[1].split('.')[0]
                if num_part.isdigit():
                    numbers.append(int(num_part))
            except:
                pass
        next_number = max(numbers) + 1 if numbers else 1
    else:
        next_number = 1
    
    renamed_count = 0
    verified_count = 0
    
    for filename in incorrectly_named:
        old_path = os.path.join(real_voices_dir, filename)
        
        # Skip if it's a directory
        if os.path.isdir(old_path):
            continue
            
        try:
            # Verify it's a valid audio file
            audio, sr = librosa.load(old_path, sr=16000)
            
            # Normalize to 4 seconds if needed
            target_samples = 4 * 16000
            if len(audio) != target_samples:
                if len(audio) > target_samples:
                    audio = audio[:target_samples]
                else:
                    padding = target_samples - len(audio)
                    audio = np.pad(audio, (0, padding))
                
                # Save normalized version
                sf.write(old_path, audio, 16000)
                print(f"📏 Normalized: {filename} to 4 seconds")
            
            # Rename file
            new_filename = f"real_{next_number:04d}.wav"
            new_path = os.path.join(real_voices_dir, new_filename)
            
            os.rename(old_path, new_path)
            print(f"📝 Renamed: {filename} -> {new_filename}")
            
            next_number += 1
            renamed_count += 1
            verified_count += 1
            
        except Exception as e:
            print(f"❌ Invalid audio file {filename}: {e}")
            try:
                os.remove(old_path)
                print(f"🗑️  Deleted invalid file: {filename}")
            except:
                print(f"🔒 Cannot delete: {filename}")
    
    # Final count of valid files
    final_files = [f for f in os.listdir(real_voices_dir) 
                  if f.startswith('real_') and f.endswith('.wav') 
                  and os.path.isfile(os.path.join(real_voices_dir, f))]
    
    valid_count = 0
    for filename in final_files:
        filepath = os.path.join(real_voices_dir, filename)
        try:
            audio, sr = librosa.load(filepath, sr=16000)
            if len(audio) == 64000 and sr == 16000:  # 4 seconds at 16kHz
                valid_count += 1
        except:
            pass  # Skip invalid files
    
    print(f"\n🎉 CLEANUP COMPLETE!")
    print(f"📊 Final valid real voice files: {valid_count}")
    print(f"📝 Files renamed: {renamed_count}")
    print(f"🔄 Files converted: {converted_count}")
    print(f"📁 Directories processed: {len(directories)}")
    
    return valid_count

if __name__ == "__main__":
    print("🧹 VERIFYING AND CLEANING REAL VOICES DATASET")
    print("=" * 60)
    
    final_count = verify_and_clean_real_voices()
    
    print(f"\n✅ Dataset verification complete!")
    print(f"💾 All valid files follow: real_XXXX.wav, 4 seconds, 16kHz")
    print(f"🎯 Ready for training with {final_count} real voices")