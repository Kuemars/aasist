import os
import glob

def rename_real_voices():
    """
    Rename all real voice files to consistent naming: real_001.wav, real_002.wav, etc.
    """
    
    real_voices_path = "data/raw/clean_real"
    
    if not os.path.exists(real_voices_path):
        print(f"❌ Directory not found: {real_voices_path}")
        return
    
    # Get all WAV files
    wav_files = glob.glob(os.path.join(real_voices_path, "*.wav"))
    
    if not wav_files:
        print(f"❌ No WAV files found in {real_voices_path}")
        return
    
    print(f"📁 Found {len(wav_files)} WAV files in {real_voices_path}")
    
    # Sort files to maintain some order
    wav_files.sort()
    
    # Rename files
    for i, old_path in enumerate(wav_files):
        # New filename: real_001.wav, real_002.wav, etc.
        new_filename = f"real_{i+1:03d}.wav"
        new_path = os.path.join(real_voices_path, new_filename)
        
        # Rename file
        os.rename(old_path, new_path)
        print(f"✅ Renamed: {os.path.basename(old_path)} -> {new_filename}")
    
    print(f"\n🎉 Renaming complete! All {len(wav_files)} files now have consistent naming.")

def count_voice_types():
    """
    Count how many of each voice type we have
    """
    
    real_voices_path = "data/raw/clean_real"
    
    if not os.path.exists(real_voices_path):
        return
    
    wav_files = os.listdir(real_voices_path)
    
    # Count by prefix
    from collections import Counter
    prefixes = [f.split('_')[0] for f in wav_files if f.endswith('.wav')]
    count = Counter(prefixes)
    
    print("\n📊 Current voice distribution:")
    for prefix, num in count.items():
        print(f"   {prefix}: {num} files")

if __name__ == "__main__":
    print("🔄 Renaming real voice files to consistent format...")
    
    # Show current distribution
    count_voice_types()
    
    # Confirm before renaming
    response = input("\n❓ Rename all files to real_001.wav format? (y/n): ")
    if response.lower() in ['y', 'yes']:
        rename_real_voices()
        print("\n✅ All real voices now have consistent naming!")
        print("💡 You can now add new datasets with any naming - just run this script again.")
    else:
        print("❌ Operation cancelled.")