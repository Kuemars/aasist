# scripts/verify_training_ready.py
import os
import torch

print("🔍 VERIFYING TRAINING READINESS")
print("=" * 60)

# Check GPU
print("\n🎮 GPU Status:")
if torch.cuda.is_available():
    print(f"✅ CUDA available")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("❌ No GPU available - training will be slow!")

# Check datasets
print("\n📁 Dataset Check:")
real_path = r"D:\sk13382\DataTraining\aasist\noise_robust_detector\data\raw\clean_real"
ai_path = r"D:\sk13382\DataTraining\aasist\noise_robust_detector\data\raw\clean_ai"

for path, name in [(real_path, "Real Voices"), (ai_path, "AI Voices")]:
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if f.endswith('.wav')]
        print(f"✅ {name}: {len(files)} files")
    else:
        print(f"❌ {name}: Path not found")

# Check AASIST import
print("\n🤖 AASIST Import Check:")
try:
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    aasist_root = os.path.dirname(os.path.dirname(script_dir))
    sys.path.insert(0, aasist_root)
    
    from models.AASIST import Model
    print("✅ AASIST import successful")
    
    # Test model creation
    config = {'nb_samp': 64600, 'nb_classes': 2}
    model = Model(config)
    print("✅ Model creation successful")
    
except Exception as e:
    print(f"❌ Import failed: {e}")

# Memory check
print("\n💾 Memory Check:")
if torch.cuda.is_available():
    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    print(f"   Free GPU memory: {free_memory / 1e9:.2f} GB")
    
    # Estimate memory needed
    # 4000 samples * 64600 * 4 bytes = ~1.03 GB for data
    # Model weights: ~10-50 MB
    # Batch size 128: ~128 * 64600 * 4 * 4 = ~132 MB per forward/backward
    estimated_memory = 1.03 + 0.05 + 0.132  # GB
    print(f"   Estimated required: {estimated_memory:.2f} GB")
    
    if free_memory / 1e9 > estimated_memory * 1.5:
        print("✅ Sufficient memory available")
    else:
        print("⚠️  Memory might be tight")

print("\n" + "=" * 60)
print("🎯 READY FOR TRAINING!")
print("Run: python scripts/train_precise_fast.py")
print("=" * 60)