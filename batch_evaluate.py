import os
import torch
import yaml
from models.AASIST import Model

print("📊 Starting Batch Evaluation...")

# Load AASIST model
with open('config/AASIST.conf', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model_config = config['model_config']
device = torch.device('cpu')
model = Model(model_config).to(device)

# LOAD PRE-TRAINED WEIGHTS (direct state dict)
weights_path = "models/weights/AASIST.pth"
if os.path.exists(weights_path):
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print("✅ Loaded pre-trained weights from AASIST.pth")
else:
    print("❌ Pre-trained weights not found. Using random initialization.")

model.eval()

# Load audio processing functions from create_model.py
import librosa
import numpy as np

def load_audio(audio_path):
    """Load and preprocess audio file for AASIST"""
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        target_length = 64600
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding))
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        return audio_tensor
    except Exception as e:
        print(f"❌ Error loading {audio_path}: {e}")
        return None

def predict_audio(audio_path):
    """Predict if audio is real or AI"""
    audio_tensor = load_audio(audio_path)
    if audio_tensor is None:
        return "ERROR", 0.0
    
    with torch.no_grad():
        output_tuple = model(audio_tensor)
        scores = output_tuple[1]
        probabilities = torch.softmax(scores, dim=1)
        human_prob = probabilities[0][0].item()
        ai_prob = probabilities[0][1].item()
        
        if ai_prob > human_prob:
            return "AI_GENERATED", ai_prob
        else:
            return "HUMAN", human_prob

print("✅ Model loaded. Starting evaluation...")

# Evaluate AI voices
print("\n🤖 Evaluating AI Voices (15 files):")
print("-" * 50)
ai_correct = 0
for i in range(1, 16):
    filename = f"ai_{i:03d}.wav"
    filepath = os.path.join('data/ai', filename)
    result, confidence = predict_audio(filepath)
    is_correct = result == "AI_GENERATED"
    if is_correct:
        ai_correct += 1
    print(f"📄 {filename}: {result} (confidence: {confidence:.3f}) {'✅' if is_correct else '❌'}")

# Evaluate Real voices  
print("\n👤 Evaluating Real Voices (15 files):")
print("-" * 50)
real_correct = 0
for i in range(1, 16):
    filename = f"real_{i:03d}.wav"
    filepath = os.path.join('data/real', filename)
    result, confidence = predict_audio(filepath)
    is_correct = result == "HUMAN"
    if is_correct:
        real_correct += 1
    print(f"📄 {filename}: {result} (confidence: {confidence:.3f}) {'✅' if is_correct else '❌'}")

# Calculate metrics
total_accuracy = (ai_correct + real_correct) / 30
ai_accuracy = ai_correct / 15
real_accuracy = real_correct / 15

print(f"\n📈 RESULTS:")
print(f"AI Detection Accuracy: {ai_accuracy:.1%} ({ai_correct}/15)")
print(f"Real Detection Accuracy: {real_accuracy:.1%} ({real_correct}/15)")
print(f"Overall Accuracy: {total_accuracy:.1%} ({ai_correct + real_correct}/30)")