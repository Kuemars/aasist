# save as debug_model.py
import os
import sys
import torch
import librosa
import numpy as np

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
from models.RawNet2Spoof import Model

# Load model
device = torch.device('cuda')
model_path = "models/rawnet2_v1.pth"

# Model config
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
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("🔍 DEBUG: What does the model output?")
print("=" * 40)

# Test with a random input
dummy = torch.randn(1, 64600).to(device)
with torch.no_grad():
    output = model(dummy)
    print(f"Model output tuple length: {len(output)}")
    print(f"First element (hidden) shape: {output[0].shape}")
    print(f"Second element (log_probs) shape: {output[1].shape}")
    print(f"\nLog probabilities (second element):")
    print(f"  Class 0: {output[1][0, 0].item():.4f}")
    print(f"  Class 1: {output[1][0, 1].item():.4f}")
    
    # Which is higher (less negative)?
    if output[1][0, 0] > output[1][0, 1]:
        print("  ➡️ Model predicts: Class 0")
    else:
        print("  ➡️ Model predicts: Class 1")
    
    # Convert to probabilities
    probs = torch.exp(output[1])
    print(f"\nProbabilities (after exp):")
    print(f"  Class 0: {probs[0, 0].item():.4f}")
    print(f"  Class 1: {probs[0, 1].item():.4f}")

# Add to debug_model.py
def test_known_sample(file_path, expected_class):
    audio, sr = librosa.load(file_path, sr=16000, duration=4.04)
    
    # Process like training
    if len(audio) > 64600:
        audio = audio[:64600]
    elif len(audio) < 64600:
        audio = np.pad(audio, (0, 64600 - len(audio)))
    
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(audio_tensor)
        log_probs = output[1]
        probs = torch.exp(log_probs)
        
    print(f"\n📄 {os.path.basename(file_path)} (expected: {expected_class}):")
    print(f"  Class 0: {probs[0,0].item():.4f}")
    print(f"  Class 1: {probs[0,1].item():.4f}")
    
    if probs[0,0] > probs[0,1]:
        prediction = "Class 0"
    else:
        prediction = "Class 1"
    
    print(f"  ➡️ Prediction: {prediction}")

# Test known files
print("\n" + "="*40)
print("Testing known samples:")
print("="*40)

# Test one real and one AI from your dataset
real_sample = "data/raw/clean_real/real_0001.wav"
ai_sample = "data/raw/clean_ai/ai_0000.wav"

if os.path.exists(real_sample):
    test_known_sample(real_sample, "HUMAN")

if os.path.exists(ai_sample):
    test_known_sample(ai_sample, "AI")
