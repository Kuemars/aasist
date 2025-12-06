import os
import sys
import torch
import librosa
import numpy as np
import json

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from models.RawNet2Spoof import Model

# Config
SAMPLE_RATE = 16000
AUDIO_LENGTH = 64600

def get_model_config():
    config_path = os.path.join(project_root, "config/RawNet2_baseline.conf")
    default_config = {
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
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if "model_config" in config:
                    model_config = config["model_config"]
                    model_config["nb_classes"] = 2
                    model_config["nb_samp"] = 64600
                    return model_config
        except:
            pass
    return default_config

def load_and_preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=4.04)
    
    if len(audio) > AUDIO_LENGTH:
        audio = audio[:AUDIO_LENGTH]
    elif len(audio) < AUDIO_LENGTH:
        audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)))
    
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return torch.FloatTensor(audio).unsqueeze(0)

def diagnose_model():
    print("🔍 MODEL DIAGNOSIS")
    print("=" * 60)
    
    # Load model
    device = torch.device('cuda')
    d_args = get_model_config()
    model = Model(d_args).to(device)
    
    # Load trained weights
    model_path = "models/aasist_best.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded: {model_path}")
    
    print("\n1. Testing random input:")
    test_tensor = torch.randn(1, 64600).to(device)
    with torch.no_grad():
        output = model(test_tensor)
        log_probs = output[1]
        print(f"   Log probabilities: {log_probs}")
        probs = torch.exp(log_probs)
        print(f"   Probabilities: {probs}")
        print(f"   Class 0 (Real?): {probs[0,0].item():.4f}")
        print(f"   Class 1 (AI?): {probs[0,1].item():.4f}")
    
    print("\n2. Testing training samples (if available):")
    
    # Test with known files from dataset
    test_files = []
    
    # Look for real sample
    real_path = "data/raw/clean_real"
    if os.path.exists(real_path):
        real_files = [os.path.join(real_path, f) for f in os.listdir(real_path) if f.endswith('.wav')]
        if real_files:
            test_files.append((real_files[0], "REAL (expected)"))
    
    # Look for AI sample  
    ai_path = "data/raw/clean_ai"
    if os.path.exists(ai_path):
        ai_files = [os.path.join(ai_path, f) for f in os.listdir(ai_path) if f.endswith('.wav')]
        if ai_files:
            test_files.append((ai_files[0], "AI (expected)"))
    
    for file_path, expected in test_files:
        try:
            audio_tensor = load_and_preprocess_audio(file_path).to(device)
            with torch.no_grad():
                output = model(audio_tensor)
                log_probs = output[1]
                probs = torch.exp(log_probs)
                
            filename = os.path.basename(file_path)
            print(f"\n   📄 {filename} - {expected}:")
            print(f"      Class 0: {probs[0,0].item():.4f}")
            print(f"      Class 1: {probs[0,1].item():.4f}")
            
            # Determine which class is higher
            if probs[0,0] > probs[0,1]:
                prediction = "Class 0"
            else:
                prediction = "Class 1"
            print(f"      Prediction: {prediction}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n3. Testing test_demo files:")
    test_demo_path = "test_demo"
    if os.path.exists(test_demo_path):
        demo_files = [os.path.join(test_demo_path, f) for f in os.listdir(test_demo_path) 
                     if f.endswith('.wav') or f.endswith('.mp3')][:3]
        
        for file_path in demo_files:
            try:
                audio_tensor = load_and_preprocess_audio(file_path).to(device)
                with torch.no_grad():
                    output = model(audio_tensor)
                    log_probs = output[1]
                    probs = torch.exp(log_probs)
                
                filename = os.path.basename(file_path)
                print(f"\n   📄 {filename}:")
                print(f"      Class 0: {probs[0,0].item():.4f}")
                print(f"      Class 1: {probs[0,1].item():.4f}")
                
                if "ai" in filename.lower():
                    print(f"      File suggests: AI")
                elif "real" in filename.lower():
                    print(f"      File suggests: REAL")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    print("\n4. Model bias check:")
    with torch.no_grad():
        # Test multiple random inputs
        total_class0 = 0
        total_class1 = 0
        for i in range(10):
            test_tensor = torch.randn(1, 64600).to(device)
            output = model(test_tensor)
            probs = torch.exp(output[1])
            if probs[0,0] > probs[0,1]:
                total_class0 += 1
            else:
                total_class1 += 1
        
        print(f"   Random inputs predicted as Class 0: {total_class0}/10")
        print(f"   Random inputs predicted as Class 1: {total_class1}/10")
        
        if total_class0 == 10:
            print("   ⚠️  MODEL IS HEAVILY BIASED TOWARD CLASS 0")
        elif total_class1 == 10:
            print("   ⚠️  MODEL IS HEAVILY BIASED TOWARD CLASS 1")
    
    print("\n=" * 60)
    print("CONCLUSION:")
    print("If model always predicts Class 0, labels might be swapped.")
    print("Try swapping class interpretation in detection script.")
    print("Or the model learned wrong labels during training.")

if __name__ == "__main__":
    diagnose_model()