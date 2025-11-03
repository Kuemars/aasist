import torch

weights_path = "models/weights/AASIST.pth"
checkpoint = torch.load(weights_path, map_location='cpu')

print("🔍 Checking weights file structure:")
print(f"Keys in checkpoint: {list(checkpoint.keys())}")

if 'model' in checkpoint:
    print("✅ 'model' key found")
    print(f"Model keys: {list(checkpoint['model'].keys())[:5]}...")  # First 5 keys
else:
    print("❌ 'model' key not found")
    # Try direct loading
    print("Attempting direct state dict loading...")
    try:
        # Check if it's already a state dict
        model_keys = list(checkpoint.keys())[:5]
        print(f"Direct keys: {model_keys}...")
    except:
        print("Could not read structure")