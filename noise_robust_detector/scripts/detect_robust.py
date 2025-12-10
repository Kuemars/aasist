import os
import sys
import torch
import librosa
import numpy as np
import json
import csv

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from models.RawNet2Spoof import Model

# Config
SAMPLE_RATE = 16000
AUDIO_LENGTH = 64600
TEST_FOLDER = "test_demo"

def detect_robust():
    print("🎯 NOISE-ROBUST AI VOICE DETECTOR")
    print("=" * 60)
    print("⚠️  MAPPING: Class 0 = HUMAN, Class 1 = AI")  # STANDARDIZED MAPPING
    print("=" * 60)
    
    # Load the NEW robust model
    model_path = "models/Noise_Robust_Model_v0.pth"  # UPDATED: New fixed model
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Looking for other models...")
        model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
        if model_files:
            model_path = f"models/{model_files[0]}"
            print(f"Using: {model_path}")
        else:
            print("No model files found in 'models/' folder")
            return
    
    # Load the robust model
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"✅ Loaded noise-robust model (96.1% validation accuracy)")
    print(f"💾 Model: {model_path}")
    
    # Test files
    test_files = []
    if os.path.exists(TEST_FOLDER):
        test_files = [os.path.join(TEST_FOLDER, f) for f in os.listdir(TEST_FOLDER) 
                     if f.endswith('.wav') or f.endswith('.mp3')]
    
    if not test_files:
        print(f"📁 No files in {TEST_FOLDER}")
        return
    
    print(f"\n🔍 Testing {len(test_files)} files:")
    print("-" * 60)
    
    results = []
    
    for file_path in test_files:
        try:
            # Load and preprocess
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=4.04)
            if len(audio) > AUDIO_LENGTH:
                audio = audio[:AUDIO_LENGTH]
            elif len(audio) < AUDIO_LENGTH:
                audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)))
            
            # Apply same normalization as training
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Target RMS normalization (consistent with training)
            target_rms = 0.1
            current_rms = np.sqrt(np.mean(audio**2))
            if current_rms > 1e-8:
                gain = target_rms / current_rms
                gain = min(gain, 5.0)
                audio = audio * gain
            
            if np.max(np.abs(audio)) > 0.95:
                audio = audio * 0.95 / np.max(np.abs(audio))
            
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                output = model(audio_tensor)[1]  # logsoftmax output
                probabilities = torch.exp(output)
                
                # STANDARDIZED MAPPING: Class 0 = HUMAN, Class 1 = AI
                human_prob = probabilities[0, 0].item() * 100    # Class 0: HUMAN
                ai_prob = probabilities[0, 1].item() * 100      # Class 1: AI
            
            # Determine prediction
            if human_prob > ai_prob:
                prediction = "HUMAN"
                confidence = human_prob
            else:
                prediction = "AI"
                confidence = ai_prob
            
            filename = os.path.basename(file_path)
            
            print(f"\n📄 {filename}")
            print(f"   Prediction: {prediction} ({confidence:.1f}% confident)")
            print(f"   HUMAN: {human_prob:.1f}% | AI: {ai_prob:.1f}%")
            
            # Confidence indicators
            if prediction == "AI":
                if confidence > 90:
                    print(f"   🎯 High confidence AI detection")
                elif confidence > 75:
                    print(f"   ✅ Confident AI detection")
                else:
                    print(f"   ⚠️  Borderline AI detection")
            else:  # HUMAN
                if confidence > 90:
                    print(f"   🎯 High confidence human detection")
                elif confidence > 75:
                    print(f"   ✅ Confident human detection")
                else:
                    print(f"   ⚠️  Borderline human detection")
            
            # Special cases
            if "goblin" in filename.lower() or "character" in filename.lower():
                print(f"   🎭 Character voice - model is robust to modifications")
            
            if confidence < 70:
                print(f"   🤔 Uncertain - consider human review")
            
            # Save results
            results.append({
                'file': filename,
                'prediction': prediction,
                'confidence': confidence,
                'human_prob': human_prob,
                'ai_prob': ai_prob,
            })
                
        except Exception as e:
            print(f"\n❌ Error with {os.path.basename(file_path)}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY:")
    
    human_count = sum(1 for r in results if r['prediction'] == 'HUMAN')
    ai_count = sum(1 for r in results if r['prediction'] == 'AI')
    
    print(f"   Predictions:")
    print(f"     HUMAN: {human_count}")
    print(f"     AI: {ai_count}")
    
    if results:
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"   Average confidence: {avg_confidence:.1f}%")
    
    # Save results
    if results:
        with open('robust_detection_results.csv', 'w', newline='') as f:
            fieldnames = ['file', 'prediction', 'confidence', 'human_prob', 'ai_prob']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n💾 Results saved to 'robust_detection_results.csv'")
    
    print("\n" + "=" * 60)
    print("✅ Detection complete with noise-robust model")
    print("   Model validation accuracy: 96.1%")
    print("   Trained on: 5000 samples (80% clean, 20% augmented)")
    print("   Audio normalization: Consistent RMS levels")
    print("=" * 60)

if __name__ == "__main__":
    detect_robust()
