# Save as test_gpu.py
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
test = torch.randn(128, 64600).cuda()
print("✅ GPU works")