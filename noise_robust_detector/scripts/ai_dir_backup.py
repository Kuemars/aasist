# scripts/backup_ai_files.py
import os
import shutil
from datetime import datetime

def backup_ai_files():
    """Create a backup before renaming"""
    
    source_dir = r"D:\sk13382\DataTraining\aasist\noise_robust_detector\data\raw\clean_ai"
    backup_dir = r"D:\sk13382\DataTraining\aasist\noise_robust_detector\data\raw\clean_ai_backup"
    
    # Add timestamp to backup folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"{backup_dir}_{timestamp}"
    
    print(f"📁 Creating backup...")
    print(f"   From: {source_dir}")
    print(f"   To: {backup_dir}")
    
    if not os.path.exists(source_dir):
        print(f"❌ Source directory doesn't exist: {source_dir}")
        return
    
    try:
        # Copy entire directory
        shutil.copytree(source_dir, backup_dir)
        print(f"✅ Backup created successfully!")
        print(f"   Backup location: {backup_dir}")
        
        # Count files
        files = [f for f in os.listdir(backup_dir) if f.endswith('.wav')]
        print(f"   Files backed up: {len(files)}")
        
    except Exception as e:
        print(f"❌ Backup failed: {e}")

if __name__ == "__main__":
    backup_ai_files()