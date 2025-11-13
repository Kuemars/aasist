import os

def print_tree(start_path, prefix="", max_depth=3, current_depth=0):
    """Print directory tree structure"""
    if current_depth > max_depth:
        return
        
    items = []
    if os.path.exists(start_path):
        items = sorted(os.listdir(start_path))
    
    for i, item in enumerate(items):
        path = os.path.join(start_path, item)
        is_last = (i == len(items) - 1)
        
        if os.path.isdir(path):
            print(f"{prefix}{'└── ' if is_last else '├── '}📁 {item}/")
            extension = "    " if is_last else "│   "
            print_tree(path, prefix + extension, max_depth, current_depth + 1)
        else:
            print(f"{prefix}{'└── ' if is_last else '├── '}📄 {item}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Project Tree: {project_root}")
    print_tree(project_root, max_depth=4)