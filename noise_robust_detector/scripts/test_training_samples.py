import os
import sys
import torch
import librosa
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.AASIST import Model
import yaml

def test_training_samples():
    """Test the model on samples from our training data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config/AASIST.conf')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model = Model(config['model_config']).to(device)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models/aasist_fine_tuned.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Test a few real voices from training data
    real_path = "data/raw/clean_real"
    real_files = [os.path.join(real_path, f) for f in os.listdir(real_path) if f.endswith('.wav')][:5]
    
    print("Testing on training real voices:")
    for file in real_files:
        audio, sr = librosa.load(file, sr=16000)
        if len(audio) > 64600:
            audio = audio[:64600]
        else:
            audio = np.pad(audio, (0, 64600 - len(audio)))
        
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output_tuple = model(audio_tensor)
            scores = output_tuple[1]
            probabilities = torch.softmax(scores, dim=1)
            human_prob = probabilities[0][0].item() * 100
            ai_prob = probabilities[0][1].item() * 100
            
            result = "HUMAN" if human_prob > ai_prob else "AI"
            print(f"  {os.path.basename(file)}: {result} (Human: {human_prob:.1f}%, AI: {ai_prob:.1f}%)")

if __name__ == "__main__":
    test_training_samples()