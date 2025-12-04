# scripts/test_environment.py
import os
import sys
import torch

print("🔍 Testing Environment")
print("=" * 50)

# Check Python version
print(f"Python: {sys.version}")

# Check PyTorch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test GPU speed
    print("\n⚡ Testing GPU...")
    a = torch.randn(10000, 10000, device='cuda')
    b = torch.randn(10000, 10000, device='cuda')
    torch.cuda.synchronize()
    c = a @ b
    torch.cuda.synchronize()
    print("✅ GPU test passed")

# Check paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

print(f"\n📁 Project root: {project_root}")
print(f"📁 Script dir: {script_dir}")

# Check if models exist
models_dir = os.path.join(project_root, "models")
if os.path.exists(models_dir):
    print(f"✅ Models directory exists")
    print("Contents:")
    for item in os.listdir(models_dir):
        if item.endswith('.py'):
            print(f"  - {item}")
else:
    print(f"❌ Models directory not found")

# Check if config exists
config_path = os.path.join(project_root, "config", "AASIST.conf")
if os.path.exists(config_path):
    print(f"✅ Config file exists: {config_path}")
else:
    print(f"❌ Config file not found")

# Check dataset paths
real_path = r"D:\sk13382\DataTraining\aasist\noise_robust_detector\data\raw\clean_real"
ai_path = r"D:\sk13382\DataTraining\aasist\noise_robust_detector\data\raw\clean_ai"

print(f"\n📁 Checking datasets...")
for path, name in [(real_path, "Real"), (ai_path, "AI")]:
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if f.endswith('.wav')]
        print(f"✅ {name}: {len(files)} files")
        if files:
            print(f"  Sample: {files[0]}")
    else:
        print(f"❌ {name} path not found: {path}")

print("\n" + "=" * 50)
print("✅ Environment check complete")