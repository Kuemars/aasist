import os
import sys
import torch
import librosa
import numpy as np
import yaml
from pathlib import Path

# Add paths for AASIST import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.AASIST import Model

class AASISTDetector:
    def __init__(self, model_path="models/aasist_fine_tuned.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Initializing AASIST Detector on {self.device}")
        
        # Load config
        config_path = "../config/AASIST.conf"
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.model_config = config['model_config']
        self.model = Model(self.model_config).to(self.device)
        
        # Load fine-tuned weights
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("✅ Loaded fine-tuned model weights")
        else:
            print("❌ Fine-tuned weights not found. Using pre-trained weights.")
            weights_path = "../models/weights/AASIST.pth"
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        self.model.eval()
        print("✅ AASIST Detector ready!")
    
    def load_audio(self, audio_path):
        """Load and preprocess audio file"""
        try:
            # Load audio at 16kHz
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # AASIST expects exactly 64600 samples
            target_length = 64600
            if len(audio) > target_length:
                audio = audio[:target_length]
            else:
                padding = target_length - len(audio)
                audio = np.pad(audio, (0, padding))
            
            return audio
        except Exception as e:
            print(f"❌ Error loading {audio_path}: {e}")
            return None
    
    def predict(self, audio_path):
        """Predict if audio is real or AI with confidence scores"""
        audio = self.load_audio(audio_path)
        if audio is None:
            return None
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output_tuple = self.model(audio_tensor)
            scores = output_tuple[1]  # [1, 2] tensor
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(scores, dim=1)
            human_prob = probabilities[0][0].item() * 100
            ai_prob = probabilities[0][1].item() * 100
            
            # Determine result
            if ai_prob > human_prob:
                result = "AI_GENERATED"
                confidence = ai_prob
            else:
                result = "HUMAN"
                confidence = human_prob
            
            return {
                'result': result,
                'confidence': confidence,
                'human_prob': human_prob,
                'ai_prob': ai_prob,
                'file_name': os.path.basename(audio_path)
            }

def create_test_directory():
    """Create a test directory structure"""
    test_dir = "test_demo"
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"\n📁 Created test directory: {test_dir}")
    print("💡 Please add your test files to this directory:")
    print(f"   - Real voice: {test_dir}/real_sample.wav")
    print(f"   - AI voice: {test_dir}/ai_sample.wav")
    print("   - Or any other audio files you want to test")
    
    return test_dir

def run_demo():
    """Run the demonstration"""
    print("=" * 60)
    print("🎯 AI VOICE DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize detector
    detector = AASISTDetector()
    
    # Create test directory
    test_dir = create_test_directory()
    
    # Check for test files
    test_files = []
    for file in os.listdir(test_dir):
        if file.endswith(('.wav', '.flac', '.mp3')):
            test_files.append(os.path.join(test_dir, file))
    
    if not test_files:
        print("\n❌ No audio files found in test directory.")
        print("Please add some audio files to the 'test_demo' folder and run again.")
        return
    
    print(f"\n🔍 Found {len(test_files)} test files:")
    for file in test_files:
        print(f"   - {os.path.basename(file)}")
    
    print("\n" + "=" * 60)
    print("📊 RUNNING DETECTION...")
    print("=" * 60)
    
    results = []
    for audio_file in test_files:
        print(f"\n🎵 Analyzing: {os.path.basename(audio_file)}")
        
        prediction = detector.predict(audio_file)
        if prediction:
            results.append(prediction)
            
            # Color-coded output
            if prediction['result'] == "AI_GENERATED":
                result_color = "🔴 AI_GENERATED"
            else:
                result_color = "🟢 HUMAN"
            
            print(f"   Result: {result_color}")
            print(f"   Confidence: {prediction['confidence']:.1f}%")
            print(f"   Human Probability: {prediction['human_prob']:.1f}%")
            print(f"   AI Probability: {prediction['ai_prob']:.1f}%")
    
    print("\n" + "=" * 60)
    print("📈 SUMMARY")
    print("=" * 60)
    
    for result in results:
        status = "🔴 AI" if result['result'] == "AI_GENERATED" else "🟢 HUMAN"
        print(f"{status}: {result['file_name']} ({result['confidence']:.1f}% confidence)")
    
    print(f"\n✅ Demonstration complete! Model successfully analyzed {len(results)} files.")

if __name__ == "__main__":
    run_demo()