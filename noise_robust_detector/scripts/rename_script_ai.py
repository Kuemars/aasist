# scripts/simple_sequential.py
import os

def simple_sequential():
    """Super simple - just rename files in the order they appear"""
    
    ai_dir = r"D:\sk13382\DataTraining\aasist\noise_robust_detector\data\raw\clean_ai"
    
    print("Super Simple Sequential Renamer")
    print("=" * 40)
    
    if not os.path.exists(ai_dir):
        print("Directory doesn't exist")
        return
    
    # Get files
    files = os.listdir(ai_dir)
    wav_files = [f for f in files if f.lower().endswith('.wav')]
    
    print(f"Found {len(wav_files)} WAV files")
    
    if not wav_files:
        print("No files to rename")
        return
    
    print("\nFirst 5 files:", wav_files[:5])
    
    # Ask for confirmation
    response = input(f"\nRename ALL {len(wav_files)} files to ai_0000.wav, ai_0001.wav, etc.? (y/n): ").strip().lower()
    
    if response != 'y':
        print("Cancelled")
        return
    
    # Do it in two passes to avoid conflicts
    print("\nStarting rename...")
    
    # Pass 1: Add .tmp extension to avoid conflicts
    temp_files = []
    for i, old_name in enumerate(wav_files):
        old_path = os.path.join(ai_dir, old_name)
        temp_name = f"temp_{i:05d}.wav"
        temp_path = os.path.join(ai_dir, temp_name)
        
        try:
            os.rename(old_path, temp_path)
            temp_files.append(temp_name)
            if i < 5:
                print(f"  {old_name} → {temp_name}")
        except Exception as e:
            print(f"  Error: {old_name} → {e}")
    
    print(f"\nPass 1 complete: {len(temp_files)} files renamed to temp")
    
    # Pass 2: Rename to final names
    print("\nFinal renaming...")
    final_count = 0
    
    for i, temp_name in enumerate(temp_files):
        temp_path = os.path.join(ai_dir, temp_name)
        final_name = f"ai_{i:04d}.wav"
        final_path = os.path.join(ai_dir, final_name)
        
        try:
            os.rename(temp_path, final_path)
            final_count += 1
            if i < 5:
                print(f"  {temp_name} → {final_name}")
        except Exception as e:
            print(f"  Error: {temp_name} → {e}")
    
    print(f"\n✅ Done! Renamed {final_count} files")
    
    # Verify
    final_files = sorted([f for f in os.listdir(ai_dir) if f.startswith('ai_')])
    print(f"\nFinal count: {len(final_files)} files")
    
    if final_files:
        print("First:", final_files[0])
        print("Last:", final_files[-1])
        
        # Quick check
        expected_first = "ai_0000.wav"
        expected_last = f"ai_{len(final_files)-1:04d}.wav"
        
        if final_files[0] == expected_first and final_files[-1] == expected_last:
            print("✅ Sequence looks correct!")
        else:
            print("⚠️  Sequence mismatch")

if __name__ == "__main__":
    simple_sequential()