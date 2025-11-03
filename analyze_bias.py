import librosa
import numpy as np
from pathlib import Path
import torch
import yaml
import sys
import os

# Add the current directory to Python path
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
        
        # Load pre-trained weights
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
            # Load audio at 16kHz
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # AASIST expects exactly 64600 samples
            target_length = 64600
            if len(audio) > target_length:
                audio = audio[:target_length]
            else:
                # Pad with zeros if too short
                padding = target_length - len(audio)
                audio = np.pad(audio, (0, padding))
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # Add batch dimension
            return audio_tensor
            
        except Exception as e:
            print(f"❌ Error loading {audio_path}: {e}")
            return None

except Exception as e:
    print(f"❌ Model loading failed: {e}")
    load_model = None
    preprocess_audio = None

def analyze_audio_differences():
    """Compare the AI vs Real voice characteristics"""
    print("=== ANALYZING AUDIO DIFFERENCES ===")
    
    # Check normalized data properties
    for voice_type in ['ai', 'real']:
        folder = Path(f"data_normalized/{voice_type}")
        files = list(folder.glob("*.wav"))
        
        print(f"\n--- {voice_type.upper()} VOICES (NORMALIZED) ---")
        print(f"Number of files: {len(files)}")
        
        durations = []
        for file_path in files[:3]:  # Check first 3 files
            audio, sr = librosa.load(file_path, sr=None)
            duration = len(audio) / sr
            durations.append(duration)
            print(f"  {file_path.name}: {duration:.2f}s, {sr}Hz, {len(audio)} samples")
        
        if durations:
            print(f"Average duration: {np.mean(durations):.2f}s")

def test_normalized_data():
    """Test the model with normalized duration data"""
    print("\n=== TESTING WITH NORMALIZED DATA ===")
    
    if load_model is None:
        print("❌ Cannot test - model loading failed")
        return
    
    # Load the model
    model = load_model()
    print("✅ Model loaded")
    
    # Test with normalized data
    data_folder = Path("data_normalized")
    
    results = {'ai': [], 'real': []}
    confidences = {'ai': [], 'real': []}
    
    for voice_type in ['ai', 'real']:
        folder = data_folder / voice_type
        files = list(folder.glob("*.wav"))
        
        print(f"\n--- Testing {voice_type} voices (NORMALIZED) ---")
        
        for file_path in files[:5]:  # Test first 5 files of each type
            try:
                # Preprocess audio
                audio_tensor = preprocess_audio(str(file_path))
                if audio_tensor is None:
                    continue
                
                # Run inference
                with torch.no_grad():
                    output_tuple = model(audio_tensor)
                    scores = output_tuple[1]  # This is the [1, 2] tensor
                    
                    # Apply softmax to get probabilities
                    probabilities = torch.softmax(scores, dim=1)
                    ai_prob = probabilities[0][1].item()
                
                # AI probability > 0.5 means AI, <= 0.5 means real
                is_ai_predicted = ai_prob > 0.5
                results[voice_type].append(is_ai_predicted)
                confidences[voice_type].append(ai_prob)
                
                print(f"  {file_path.name}: {ai_prob:.3f} -> {'AI' if is_ai_predicted else 'Real'}")
                
            except Exception as e:
                print(f"  ❌ Error with {file_path.name}: {e}")
    
    # Calculate accuracy
    ai_correct = sum(results['ai'])  # AI files correctly identified as AI
    real_correct = len(results['real']) - sum(results['real'])  # Real files correctly identified as Real
    
    total_tested = len(results['ai']) + len(results['real'])
    accuracy = (ai_correct + real_correct) / total_tested if total_tested > 0 else 0
    
    print(f"\n📊 RESULTS WITH NORMALIZED DATA:")
    print(f"AI detection: {ai_correct}/{len(results['ai'])} ({ai_correct/len(results['ai'])*100:.1f}%)")
    print(f"Real detection: {real_correct}/{len(results['real'])} ({real_correct/len(results['real'])*100:.1f}%)")
    print(f"Overall accuracy: {accuracy*100:.1f}%")
    
    # Show confidence analysis
    if confidences['ai'] and confidences['real']:
        print(f"\n🎯 CONFIDENCE ANALYSIS:")
        print(f"AI voices - Avg confidence: {np.mean(confidences['ai']):.3f}")
        print(f"Real voices - Avg confidence: {np.mean(confidences['real']):.3f}")
    
    return accuracy

if __name__ == "__main__":
    analyze_audio_differences()
    test_normalized_data()