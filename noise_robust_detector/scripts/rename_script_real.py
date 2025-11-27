import os
import re

def fix_padding():
    """
    Fix inconsistent padding in real_*.wav filenames
    Changes real_043.wav → real_0043.wav, real_201.wav → real_0201.wav, etc.
    """
    
    real_voices_dir = "data/raw/clean_real"
    
    if not os.path.exists(real_voices_dir):
        print(f"❌ Directory not found: {real_voices_dir}")
        return
    
    # Get all real_*.wav files
    files = [f for f in os.listdir(real_voices_dir) if f.startswith('real_') and f.endswith('.wav')]
    
    if not files:
        print(f"❌ No real_*.wav files found in {real_voices_dir}")
        return
    
    print(f"📁 Found {len(files)} real_*.wav files")
    
    # Find files with inconsistent padding
    files_to_rename = []
    
    for filename in files:
        match = re.match(r'real_(\d+)\.wav', filename)
        if match:
            number = int(match.group(1))
            current_padding = len(match.group(1))
            
            # Check if padding is inconsistent (not 4 digits)
            if current_padding != 4:
                new_filename = f"real_{number:04d}.wav"
                files_to_rename.append((filename, new_filename))
    
    if not files_to_rename:
        print("✅ All files already have consistent 4-digit padding!")
        return
    
    print(f"🔄 Fixing padding for {len(files_to_rename)} files...")
    
    # Rename files
    renamed_count = 0
    
    for old_filename, new_filename in files_to_rename:
        old_path = os.path.join(real_voices_dir, old_filename)
        new_path = os.path.join(real_voices_dir, new_filename)
        
        try:
            # Check if new filename already exists (shouldn't happen with proper numbering)
            if os.path.exists(new_path):
                print(f"⚠️  Skipping {old_filename} → {new_filename} (target exists)")
                continue
                
            os.rename(old_path, new_path)
            print(f"📝 {old_filename} → {new_filename}")
            renamed_count += 1
            
        except Exception as e:
            print(f"❌ Error renaming {old_filename}: {e}")
    
    print(f"\n🎉 PADDING FIXED!")
    print(f"📊 Successfully renamed {renamed_count} files")
    
    # Show final file pattern
    remaining_files = [f for f in os.listdir(real_voices_dir) if f.startswith('real_') and f.endswith('.wav')]
    if remaining_files:
        sample_files = sorted(remaining_files)[:5]
        print(f"📝 Sample files: {', '.join(sample_files)}")

def check_current_padding():
    """
    Check the current padding pattern of files
    """
    
    real_voices_dir = "data/raw/clean_real"
    files = [f for f in os.listdir(real_voices_dir) if f.startswith('real_') and f.endswith('.wav')]
    
    if not files:
        print("❌ No files found")
        return
    
    # Analyze padding patterns
    padding_counts = {}
    
    for filename in files:
        match = re.match(r'real_(\d+)\.wav', filename)
        if match:
            padding = len(match.group(1))
            padding_counts[padding] = padding_counts.get(padding, 0) + 1
    
    print("📊 CURRENT PADDING ANALYSIS:")
    for padding, count in sorted(padding_counts.items()):
        print(f"   {padding}-digit padding: {count} files")
    
    # Show examples of each padding type
    print(f"\n🔍 Examples:")
    for padding in sorted(padding_counts.keys()):
        examples = [f for f in files if re.match(fr'real_\d{{{padding}}}\.wav', f)]
        if examples:
            sample = sorted(examples)[:3]
            print(f"   {padding}-digit: {', '.join(sample)}")

if __name__ == "__main__":
    print("🔢 FIX FILENAME PADDING")
    print("=" * 50)
    
    print("1. Check current padding patterns")
    print("2. Fix all files to 4-digit padding")
    
    choice = input("Choose option (1 or 2): ")
    
    if choice == "1":
        check_current_padding()
    elif choice == "2":
        fix_padding()
    else:
        print("❌ Invalid choice")