import os
import sys
import torch
import librosa
import numpy as np
import json
from pathlib import Path

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
from models.AASIST import Model

# Config
SAMPLE_RATE = 16000
AUDIO_LENGTH = 64600
TEST_FOLDER = "test_demo"

def get_model_config():
    config_path = os.path.join(project_root, "config/AASIST.conf")
    with open(config_path, 'r') as f:
        config = json.load(f)
        model_config = config["model_config"]
        model_config["nb_classes"] = 2
        model_config["nb_samp"] = 64600
    return model_config

def load_and_preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=4.04)
    
    if len(audio) > AUDIO_LENGTH:
        audio = audio[:AUDIO_LENGTH]
    elif len(audio) < AUDIO_LENGTH:
        audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)))
    
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return torch.FloatTensor(audio).unsqueeze(0)

def detect_voices():
    print("🎯 AI VOICE DETECTOR - USING TRAINED MODEL")
    print("=" * 50)
    
    # Check for model
    model_path = "models/aasist_best.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Looking for other saved models...")
        model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
        if model_files:
            model_path = f"models/{model_files[0]}"
            print(f"Using: {model_path}")
        else:
            print("No model files found in 'models/' folder")
            return
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    model_config = get_model_config()
    model = Model(model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Loaded model: {model_path}")
    
    # Get test files
    test_files = []
    if os.path.exists(TEST_FOLDER):
        test_files = [os.path.join(TEST_FOLDER, f) for f in os.listdir(TEST_FOLDER) 
                     if f.endswith('.wav') or f.endswith('.mp3')]
    
    if not test_files:
        print(f"\n📁 No audio files found in '{TEST_FOLDER}/'")
        print("Please add .wav or .mp3 files to the test_demo folder")
        return
    
    print(f"\n🔍 Testing {len(test_files)} audio files:")
    print("-" * 50)
    
    results = []
    for file_path in test_files:
        try:
            audio_tensor = load_and_preprocess_audio(file_path).to(device)
            
            with torch.no_grad():
                output = model(audio_tensor)[1]
                probabilities = torch.softmax(output, dim=1)
                human_prob = probabilities[0, 0].item() * 100
                ai_prob = probabilities[0, 1].item() * 100
            
            if human_prob > ai_prob:
                prediction = "HUMAN"
                confidence = human_prob
            else:
                prediction = "AI"
                confidence = ai_prob
            
            filename = os.path.basename(file_path)
            results.append({
                'file': filename,
                'prediction': prediction,
                'confidence': confidence,
                'human_prob': human_prob,
                'ai_prob': ai_prob
            })
            
            print(f"📄 {filename}")
            print(f"   Prediction: {prediction} ({confidence:.1f}% confident)")
            print(f"   Human: {human_prob:.1f}% | AI: {ai_prob:.1f}%")
            print()
            
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(file_path)}: {e}")
    
    # Summary
    print("=" * 50)
    print("📊 SUMMARY:")
    
    human_count = sum(1 for r in results if r['prediction'] == 'HUMAN')
    ai_count = sum(1 for r in results if r['prediction'] == 'AI')
    
    print(f"   Human predictions: {human_count}")
    print(f"   AI predictions: {ai_count}")
    
    if results:
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"   Average confidence: {avg_confidence:.1f}%")
    
    # Save results
    if results:
        import csv
        with open('detection_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file', 'prediction', 'confidence', 'human_prob', 'ai_prob'])
            writer.writeheader()
            writer.writerows(results)
        print(f"\n💾 Results saved to 'detection_results.csv'")

if __name__ == "__main__":
    detect_voices()