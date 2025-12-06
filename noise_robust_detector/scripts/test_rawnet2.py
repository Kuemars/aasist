import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from models.RawNet2Spoof import Model

device = torch.device('cuda')

d_args = {
    "architecture": "RawNet2Spoof",
    "nb_samp": 64600,
    "first_conv": 1024,
    "in_channels": 1,
    "filts": [20, [20, 20], [20, 128], [128, 128]],
    "blocks": [2, 4],
    "nb_fc_node": 1024,
    "gru_node": 1024,
    "nb_gru_layer": 3,
    "nb_classes": 2
}

model = Model(d_args).to(device)

print(f"✅ RawNet2 Model loaded")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
dummy = torch.randn(2, 64600).to(device)
output = model(dummy)
print(f"Output shape: {output[1].shape}")  # Should be [2, 2]

if output[1].shape == torch.Size([2, 2]):
    print("🎯 TEST PASSED - Ready for training")
else:
    print(f"❌ TEST FAILED - Got shape {output[1].shape}")