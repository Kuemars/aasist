import librosa
import numpy as np
from pathlib import Path
import torch
import yaml
import sys
import os

sys.path.append('.')

try:
    from models.AASIST import Model
    
    def load_model():
        """Load the AASIST model with pre-trained weights"""
        with open('config/AASIST.conf', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        model_config = config['model_config']
        device = torch.device('cpu')
        model = Model(model_config).to(device)
        
        weights_path = "models/weights/AASIST.pth"
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print("✅ Loaded pre-trained weights from AASIST.pth")
        else:
            print("⚠️  Pre-trained weights not found. Using random initialization.")
        
        model.eval()
        return model
    
    def preprocess_audio(audio_path):
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

except Exception as e:
    print(f"❌ Model loading failed: {e}")
    load_model = None
    preprocess_audio = None

def test_enhanced_data():
    """Test the model with enhanced real voices"""
    print("=== TESTING WITH ENHANCED REAL VOICES ===")
    
    if load_model is None:
        print("❌ Cannot test - model loading failed")
        return
    
    model = load_model()
    print("✅ Model loaded")
    
    # Test setup: AI voices (original) vs Real voices (enhanced)
    ai_folder = Path("data_normalized/ai")
    real_folder = Path("data_enhanced/real")
    
    results = {'ai': [], 'real': []}
    confidences = {'ai': [], 'real': []}
    
    # Test AI voices (unchanged)
    print(f"\n--- Testing AI voices ---")
    ai_files = list(ai_folder.glob("*.wav"))[:5]
    for file_path in ai_files:
        try:
            audio_tensor = preprocess_audio(str(file_path))
            if audio_tensor is None:
                continue
            
            with torch.no_grad():
                output_tuple = model(audio_tensor)
                scores = output_tuple[1]
                probabilities = torch.softmax(scores, dim=1)
                ai_prob = probabilities[0][1].item()
            
            is_ai_predicted = ai_prob > 0.5
            results['ai'].append(is_ai_predicted)
            confidences['ai'].append(ai_prob)
            
            print(f"  {file_path.name}: {ai_prob:.3f} -> {'AI' if is_ai_predicted else 'Real'}")
            
        except Exception as e:
            print(f"  ❌ Error with {file_path.name}: {e}")
    
    # Test ENHANCED real voices
    print(f"\n--- Testing ENHANCED real voices ---")
    real_files = list(real_folder.glob("*.wav"))[:5]
    for file_path in real_files:
        try:
            audio_tensor = preprocess_audio(str(file_path))
            if audio_tensor is None:
                continue
            
            with torch.no_grad():
                output_tuple = model(audio_tensor)
                scores = output_tuple[1]
                probabilities = torch.softmax(scores, dim=1)
                ai_prob = probabilities[0][1].item()
            
            is_ai_predicted = ai_prob > 0.5
            results['real'].append(is_ai_predicted)
            confidences['real'].append(ai_prob)
            
            print(f"  {file_path.name}: {ai_prob:.3f} -> {'AI' if is_ai_predicted else 'Real'}")
            
        except Exception as e:
            print(f"  ❌ Error with {file_path.name}: {e}")
    
    # Calculate accuracy
    ai_correct = sum(results['ai'])
    real_correct = len(results['real']) - sum(results['real'])
    
    total_tested = len(results['ai']) + len(results['real'])
    accuracy = (ai_correct + real_correct) / total_tested if total_tested > 0 else 0
    
    print(f"\n📊 RESULTS WITH ENHANCED REAL VOICES:")
    print(f"AI detection: {ai_correct}/{len(results['ai'])} ({ai_correct/len(results['ai'])*100:.1f}%)")
    print(f"Real detection: {real_correct}/{len(results['real'])} ({real_correct/len(results['real'])*100:.1f}%)")
    print(f"Overall accuracy: {accuracy*100:.1f}%")
    
    if confidences['ai'] and confidences['real']:
        print(f"\n🎯 CONFIDENCE ANALYSIS:")
        print(f"AI voices - Avg confidence: {np.mean(confidences['ai']):.3f}")
        print(f"Real voices - Avg confidence: {np.mean(confidences['real']):.3f}")

if __name__ == "__main__":
    test_enhanced_data()