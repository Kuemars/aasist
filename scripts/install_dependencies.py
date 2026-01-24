import subprocess
import sys
import importlib

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")
        if e.stderr:
            print(f"   Error details: {e.stderr}")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def main():
    print("🚀 INSTALLING AI VOICE DETECTION DEPENDENCIES")
    print("=" * 50)
    
    # Check if we're in a conda environment
    in_conda = run_command("conda info --base", "Checking conda environment")
    
    # Install PyTorch with CUDA support
    pytorch_command = "pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118"
    run_command(pytorch_command, "Installing PyTorch with CUDA 11.8")
    
    # Install audio processing packages
    audio_packages = [
        "librosa==0.10.1",
        "soundfile==0.12.1", 
        "gtts==2.3.2"
    ]
    
    for package in audio_packages:
        run_command(f"pip install {package}", f"Installing {package}")
    
    # Install data science packages
    ds_packages = [
        "numpy==1.24.3",
        "scipy==1.10.1",
        "scikit-learn==1.3.0",
        "matplotlib==3.7.1",
        "pyyaml==6.0"
    ]
    
    for package in ds_packages:
        run_command(f"pip install {package}", f"Installing {package}")
    
    print("\n" + "=" * 50)
    print("✅ INSTALLATION COMPLETE")
    print("\n🔍 VERIFYING INSTALLATION...")
    
    # Verify critical packages
    critical_packages = {
        "torch": "torch",
        "librosa": "librosa",
        "soundfile": "soundfile",
        "gtts": "gtts",
        "numpy": "numpy",
        "scipy": "scipy",
        "sklearn": "sklearn",
        "matplotlib": "matplotlib",
        "yaml": "yaml"
    }
    
    all_good = True
    for display_name, import_name in critical_packages.items():
        if check_package(display_name, import_name):
            print(f"   ✅ {display_name:15} - OK")
        else:
            print(f"   ❌ {display_name:15} - MISSING")
            all_good = False
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA           - Available ({torch.cuda.get_device_name(0)})")
        else:
            print("   ⚠️  CUDA           - Not available (CPU only)")
    except:
        print("   ❌ CUDA           - Check failed")
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
        print("You can now run: python scripts/fine_tune_aasist.py")
    else:
        print("⚠️  Some packages failed to install. Check errors above.")

if __name__ == "__main__":
    main()
