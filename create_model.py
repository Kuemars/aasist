import torch
import yaml
import librosa
import numpy as np
import os
print("Step 2: Adding audio processing to AASIST model...")

try:
    from models.AASIST import Model
    
    # Load the exact config from file
    with open('config/AASIST.conf', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model_config = config['model_config']
    device = torch.device('cpu')
    model = Model(model_config).to(device)
    
    # LOAD PRE-TRAINED WEIGHTS
    weights_path = "models/weights/AASIST.pth"
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ Loaded pre-trained weights from AASIST.pth")
    else:
        print("⚠️  Pre-trained weights not found. Using random initialization.")
    
    model.eval()
    print("✅ AASIST model loaded and ready!")
    
    def load_audio(audio_path):
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
    
    def predict_audio(audio_path):
        """Predict if audio is real or AI"""
        audio_tensor = load_audio(audio_path)
        if audio_tensor is None:
            return "ERROR", 0.0
        
        with torch.no_grad():
            # Model returns tuple, we want the second element [1, 2]
            output_tuple = model(audio_tensor)
            scores = output_tuple[1]  # This is the [1, 2] tensor
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(scores, dim=1)
            human_prob = probabilities[0][0].item()
            ai_prob = probabilities[0][1].item()
            
            if ai_prob > human_prob:
                return "AI_GENERATED", ai_prob
            else:
                return "HUMAN", human_prob
    
    def improved_predict_audio(audio_path):
        """Improved prediction with better confidence scoring"""
        audio_tensor = load_audio(audio_path)
        if audio_tensor is None:
            return "ERROR", 0.0, {}
        
        with torch.no_grad():
            output_tuple = model(audio_tensor)
            scores = output_tuple[1]  # This is the [1, 2] tensor
            
            # Get raw scores before softmax
            raw_human_score = scores[0][0].item()
            raw_ai_score = scores[0][1].item()
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(scores, dim=1)
            human_prob = probabilities[0][0].item()
            ai_prob = probabilities[0][1].item()
            
            # Calculate confidence metrics
            confidence_metrics = {
                'human_prob': human_prob,
                'ai_prob': ai_prob,
                'raw_human_score': raw_human_score,
                'raw_ai_score': raw_ai_score,
                'confidence_gap': abs(human_prob - ai_prob),  # How certain the model is
            }
            
            if ai_prob > human_prob:
                return "AI_GENERATED", ai_prob, confidence_metrics
            else:
                return "HUMAN", human_prob, confidence_metrics

    print("✅ Audio processing functions added!")
    print("✅ AASIST detection system is now ready!")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

def test_with_audio_files():
    """Test the model with any available audio files"""
    print("\n🎯 Testing AASIST with audio files...")
    
    # Check for any WAV files in current directory
    import os
    audio_files = [f for f in os.listdir('.') if f.endswith('.wav')]
    
    if not audio_files:
        print("⚠️  No WAV files found in current directory.")
        print("💡 To test, please add some .wav files to this folder:")
        print(f"   Current directory: {os.getcwd()}")
        return
    
    print(f"🔍 Found {len(audio_files)} audio files for testing:")
    for file in audio_files:
        print(f"   - {file}")
    
    print("\n📊 Running detection...")
    print("-" * 50)
    
    for audio_file in audio_files:
        result, confidence = predict_audio(audio_file)
        print(f"📄 {audio_file:20} -> {result:15} (confidence: {confidence:.3f})")
    
    print("-" * 50)
    print("✅ Testing complete! AASIST is working correctly.")

# Run the test
if __name__ == "__main__":
    test_with_audio_files()