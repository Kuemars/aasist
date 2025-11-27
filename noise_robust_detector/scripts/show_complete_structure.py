import os

def print_tree(start_path, prefix="", max_depth=4, current_depth=0):
    """Print complete directory tree structure"""
    if current_depth > max_depth:
        return
        
    items = []
    if os.path.exists(start_path):
        try:
            items = sorted(os.listdir(start_path))
        except PermissionError:
            print(f"{prefix}🔒 [Permission Denied]")
            return
    
    # Filter and sort items
    dirs = []
    files = []
    for item in items:
        if item.startswith('.'):  # Skip hidden files
            continue
        item_path = os.path.join(start_path, item)
        if os.path.isdir(item_path):
            dirs.append(item)
        else:
            files.append(item)
    
    # Print directories first
    for i, item in enumerate(dirs):
        path = os.path.join(start_path, item)
        is_last = (i == len(dirs) - 1 and len(files) == 0)
        
        print(f"{prefix}{'└── ' if is_last else '├── '}📁 {item}/")
        extension = "    " if is_last else "│   "
        print_tree(path, prefix + extension, max_depth, current_depth + 1)
    
    # Print files
    for i, item in enumerate(files):
        is_last = (i == len(files) - 1)
        
        # Get file size
        file_path = os.path.join(start_path, item)
        try:
            size = os.path.getsize(file_path)
            size_str = f" ({size:,} bytes)"
        except:
            size_str = ""
        
        # Get file extension for emoji
        ext = os.path.splitext(item)[1].lower()
        emoji = "📄"  # Default
        if ext in ['.py']:
            emoji = "🐍"
        elif ext in ['.md', '.txt']:
            emoji = "📝"  
        elif ext in ['.wav', '.mp3', '.flac']:
            emoji = "🎵"
        elif ext in ['.pth', '.pt', '.bin']:
            emoji = "⚖️"
        elif ext in ['.json', '.yaml', '.yml', '.conf']:
            emoji = "⚙️"
        elif ext in ['.zip', '.tar', '.gz']:
            emoji = "📦"
        
        print(f"{prefix}{'└── ' if is_last else '├── '}{emoji} {item}{size_str}")

def get_folder_size(start_path):
    """Calculate total size of a folder"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for filename in filenames:
            try:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
            except:
                pass
    return total_size

def analyze_project():
    """Analyze project structure and statistics"""
    project_root = "D:/sk13382/DataTraining/aasist"
    
    print("🚀 COMPLETE PROJECT STRUCTURE ANALYSIS")
    print("=" * 70)
    print(f"📍 Project Root: {project_root}")
    print()
    
    # Main project structure
    print("📁 MAIN PROJECT STRUCTURE:")
    print_tree(project_root, max_depth=3)
    
    print("\n" + "=" * 70)
    print("📊 PROJECT STATISTICS:")
    print("=" * 70)
    
    # Count files by type
    file_types = {}
    total_files = 0
    
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.startswith('.'):
                continue
            ext = os.path.splitext(file)[1].lower()
            file_types[ext] = file_types.get(ext, 0) + 1
            total_files += 1
    
    print(f"📄 Total Files: {total_files}")
    print("\n📋 File Type Breakdown:")
    for ext, count in sorted(file_types.items()):
        if ext == '':
            ext_name = 'no extension'
        else:
            ext_name = ext
        print(f"   {ext_name:15} : {count:4d} files")
    
    # Dataset statistics
    print("\n🎵 DATASET STATISTICS:")
    real_voices_dir = "data/raw/clean_real"
    ai_voices_dir = "data/raw/clean_ai"
    
    if os.path.exists(real_voices_dir):
        real_files = [f for f in os.listdir(real_voices_dir) if f.endswith('.wav')]
        print(f"   🎤 Real Voices: {len(real_files)} files")
        if real_files:
            real_size = get_folder_size(real_voices_dir)
            print(f"      💾 Size: {real_size/(1024**2):.1f} MB")
    
    if os.path.exists(ai_voices_dir):
        ai_files = [f for f in os.listdir(ai_voices_dir) if f.endswith('.wav')]
        print(f"   🤖 AI Voices: {len(ai_files)} files")
        if ai_files:
            ai_size = get_folder_size(ai_voices_dir)
            print(f"      💾 Size: {ai_size/(1024**2):.1f} MB")
    
    # Scripts count
    scripts_dir = "noise_robust_detector/scripts"
    if os.path.exists(scripts_dir):
        script_files = [f for f in os.listdir(scripts_dir) if f.endswith('.py')]
        print(f"   🐍 Python Scripts: {len(script_files)} files")
    
    # Virtual environment
    venv_dir = "aasist_env"
    if os.path.exists(venv_dir):
        venv_size = get_folder_size(venv_dir)
        print(f"   🐍 Virtual Environment: {venv_size/(1024**2):.1f} MB")
    
    # Total project size
    total_size = get_folder_size(project_root)
    print(f"\n💾 TOTAL PROJECT SIZE: {total_size/(1024**3):.2f} GB")
    
    print("\n" + "=" * 70)
    print("🎯 PROJECT STATUS: READY FOR FINE-TUNING!")
    print("=" * 70)

if __name__ == "__main__":
    analyze_project()