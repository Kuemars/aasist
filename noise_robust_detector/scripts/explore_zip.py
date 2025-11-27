import zipfile
import os

def explore_zip(zip_path):
    """Explore what's inside the zip file"""
    print(f"🔍 Exploring zip file: {zip_path}")
    
    if not os.path.exists(zip_path):
        print(f"❌ Zip file not found: {zip_path}")
        return
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get all file names
            all_files = zip_ref.namelist()
            
            print(f"📁 Total files in zip: {len(all_files)}")
            
            # Show first 20 files to understand structure
            print("\n📋 First 20 files:")
            for i, file_name in enumerate(all_files[:20]):
                print(f"   {i+1:2d}. {file_name}")
            
            # Count file types
            file_types = {}
            for file_name in all_files:
                ext = os.path.splitext(file_name)[1].lower()
                file_types[ext] = file_types.get(ext, 0) + 1
            
            print(f"\n📊 File type breakdown:")
            for ext, count in sorted(file_types.items()):
                print(f"   {ext or 'no ext'}: {count} files")
            
            # Look for audio files
            audio_files = [f for f in all_files if f.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a'))]
            print(f"\n🎵 Audio files found: {len(audio_files)}")
            
            if audio_files:
                print("Sample audio files:")
                for i, audio_file in enumerate(audio_files[:10]):
                    print(f"   {i+1:2d}. {audio_file}")
    
    except Exception as e:
        print(f"❌ Error exploring zip: {e}")

if __name__ == "__main__":
    zip_path = "D:/sk13382/archive.zip"
    explore_zip(zip_path)