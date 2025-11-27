import os
import librosa
import soundfile as sf
import numpy as np

def identify_corrupted_files():
    """
    Identify which files are corrupted and need to be removed
    """
    
    real_voices_dir = "data/raw/clean_real"
    
    if not os.path.exists(real_voices_dir):
        print(f"❌ Directory not found: {real_voices_dir}")
        return
    
    files = [f for f in os.listdir(real_voices_dir) if f.startswith('real_') and f.endswith('.wav')]
    print(f"📁 Found {len(files)} real_*.wav files")
    
    valid_files = []
    corrupted_files = []
    wrong_format_files = []
    
    print(f"\n🔍 Checking each file...")
    
    for i, filename in enumerate(files):
        filepath = os.path.join(real_voices_dir, filename)
        
        try:
            # Try to load the file
            audio, sr = librosa.load(filepath, sr=16000)
            
            # Check format
            if len(audio) == 64000 and sr == 16000:
                valid_files.append(filename)
            else:
                wrong_format_files.append((filename, f"{len(audio)} samples, {sr} Hz"))
                
        except Exception as e:
            corrupted_files.append((filename, str(e)))
        
        # Progress update
        if (i + 1) % 100 == 0:
            print(f"   Checked {i + 1}/{len(files)} files...")
    
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"   ✅ Valid files: {len(valid_files)}")
    print(f"   ⚠️  Wrong format: {len(wrong_format_files)}")
    print(f"   ❌ Corrupted: {len(corrupted_files)}")
    
    # Show some examples of corrupted files
    if corrupted_files:
        print(f"\n🔍 Sample corrupted files:")
        for filename, error in corrupted_files[:10]:
            print(f"   {filename}: {error}")
    
    if wrong_format_files:
        print(f"\n🔍 Sample wrong format files:")
        for filename, format_info in wrong_format_files[:10]:
            print(f"   {filename}: {format_info}")
    
    # Ask if user wants to remove corrupted files
    if corrupted_files or wrong_format_files:
        response = input(f"\n❓ Remove {len(corrupted_files) + len(wrong_format_files)} corrupted/wrong-format files? (y/n): ")
        if response.lower() in ['y', 'yes']:
            removed_count = 0
            for filename, _ in corrupted_files + wrong_format_files:
                try:
                    filepath = os.path.join(real_voices_dir, filename)
                    os.remove(filepath)
                    removed_count += 1
                except Exception as e:
                    print(f"❌ Could not remove {filename}: {e}")
            
            print(f"🗑️  Removed {removed_count} files")
            
            # Re-count valid files
            remaining_files = [f for f in os.listdir(real_voices_dir) if f.startswith('real_') and f.endswith('.wav')]
            valid_remaining = 0
            for filename in remaining_files:
                filepath = os.path.join(real_voices_dir, filename)
                try:
                    audio, sr = librosa.load(filepath, sr=16000)
                    if len(audio) == 64000 and sr == 16000:
                        valid_remaining += 1
                except:
                    pass
            
            print(f"📊 Final valid files: {valid_remaining}")
        else:
            print("❌ No files removed")
    
    return valid_files

def fix_wrong_format_files():
    """
    Try to fix files with wrong format (wrong length or sample rate)
    """
    
    real_voices_dir = "data/raw/clean_real"
    files = [f for f in os.listdir(real_voices_dir) if f.startswith('real_') and f.endswith('.wav')]
    
    fixed_count = 0
    
    for filename in files:
        filepath = os.path.join(real_voices_dir, filename)
        
        try:
            audio, sr = librosa.load(filepath, sr=16000)
            
            # Fix if wrong length
            target_samples = 64000
            if len(audio) != target_samples:
                if len(audio) > target_samples:
                    audio = audio[:target_samples]
                else:
                    padding = target_samples - len(audio)
                    audio = np.pad(audio, (0, padding))
                
                # Overwrite with fixed version
                sf.write(filepath, audio, 16000)
                fixed_count += 1
                print(f"🔧 Fixed: {filename} (was {len(audio)} samples)")
                
        except Exception as e:
            # This file is corrupted, skip it
            pass
    
    print(f"🔧 Fixed {fixed_count} files with wrong format")

if __name__ == "__main__":
    print("🔍 IDENTIFYING CORRUPTED FILES")
    print("=" * 50)
    
    print("1. Identify corrupted files")
    print("2. Fix wrong format files")
    
    choice = input("Choose option (1 or 2): ")
    
    if choice == "1":
        identify_corrupted_files()
    elif choice == "2":
        fix_wrong_format_files()
    else:
        print("❌ Invalid choice")