"""
VERIFICATION SCRIPT - Step 1 Completion Check
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from project_config import verify_structure, PROJECT_ROOT

def check_python_environment():
    """Check if required packages are available"""
    print("🐍 Checking Python environment...")
    
    required_packages = [
        "torch", "torchaudio", "librosa", "numpy", 
        "soundfile", "matplotlib", "scipy"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        print("   Run: pip install " + " ".join(missing_packages))
    else:
        print("✅ All required packages installed!")

def main():
    print("=" * 60)
    print("NOISE-ROBUST AI VOICE DETECTOR - SETUP VERIFICATION")
    print("=" * 60)
    
    # Verify project structure
    verify_structure()
    
    # Check environment
    check_python_environment()
    
    print("\n" + "=" * 60)
    print("🎯 NEXT STEPS:")
    print("1. If all checks pass, proceed to data collection")
    print("2. If issues found, fix them before continuing")
    print("=" * 60)

if __name__ == "__main__":
    main()