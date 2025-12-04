import os
from pathlib import Path

def print_tree(start_path, prefix="", max_depth=3, current_depth=0, exclude_dirs=None):
    """Print directory tree structure excluding specific directories"""
    if exclude_dirs is None:
        exclude_dirs = []
    
    if current_depth > max_depth:
        return
        
    items = []
    if os.path.exists(start_path):
        items = sorted(os.listdir(start_path))
    
    # Filter out excluded directories
    filtered_items = []
    for item in items:
        if item in exclude_dirs and os.path.isdir(os.path.join(start_path, item)):
            continue
        filtered_items.append(item)
    
    for i, item in enumerate(filtered_items):
        path = os.path.join(start_path, item)
        is_last = (i == len(filtered_items) - 1)
        
        if os.path.isdir(path):
            print(f"{prefix}{'└── ' if is_last else '├── '}📁 {item}/")
            extension = "    " if is_last else "│   "
            print_tree(path, prefix + extension, max_depth, current_depth + 1, exclude_dirs)
        else:
            print(f"{prefix}{'└── ' if is_last else '├── '}📄 {item}")

if __name__ == "__main__":
    # Get the project root (two levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    noise_robust_detector_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(noise_robust_detector_dir)
    
    print(f"Project Tree: {project_root}")
    print("=" * 50)
    
    # Define directories to exclude - specifically the 'data' directory inside noise_robust_detector
    exclude_dirs = ['data']
    
    print_tree(project_root, max_depth=4, exclude_dirs=exclude_dirs)