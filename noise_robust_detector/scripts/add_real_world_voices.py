import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import numpy as np
import librosa

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.AASIST import Model
from fine_tune_aasist import AudioDataset, collate_fn, get_aasist_config
from sklearn.model_selection import train_test_split

def add_noise_to_audio(audio, noise_level=0.01):
    """Add slight noise to simulate real-world conditions"""
    noise = np.random.normal(0, noise_level, len(audio))
    return audio + noise

class RealWorldAudioDataset:
    def __init__(self, real_files, ai_files, augment_real=True):
        self.real_files = real_files
        self.ai_files = ai_files
        self.augment_real = augment_real
        
        self.files = real_files + ai_files
        self.labels = [0] * len(real_files) + [1] * len(ai_files)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Ensure correct length
        if len(audio) > 64600:
            audio = audio[:64600]
        else:
            padding = 64600 - len(audio)
            audio = np.pad(audio, (0, padding))
        
        # Add noise to REAL voices to simulate real-world conditions
        if self.augment_real and self.labels[idx] == 0:
            # Randomly apply different real-world effects
            if np.random.random() < 0.3:  # 30% chance
                audio = add_noise_to_audio(audio, noise_level=0.005)
            if np.random.random() < 0.2:  # 20% chance  
                # Simulate room reverb
                audio = audio * (0.9 + 0.1 * np.random.random())
        
        audio_tensor = torch.FloatTensor(audio)
        label_tensor = torch.LongTensor([self.labels[idx]])
        
        return audio_tensor, label_tensor

def fix_real_world_detection():
    """Fix the model to work with real-world recordings"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Fixing real-world detection on {device}")
    
    # Load model
    model_config = get_aasist_config()
    model = Model(model_config).to(device)
    model.load_state_dict(torch.load('models/aasist_fine_tuned.pth', map_location=device))
    
    # Get data
    REAL_PATH = "data/raw/clean_real"
    AI_PATH = "data/raw/clean_ai"
    
    real_files = [os.path.join(REAL_PATH, f) for f in os.listdir(REAL_PATH) if f.endswith('.wav')]
    ai_files = [os.path.join(AI_PATH, f) for f in os.listdir(AI_PATH) if f.endswith('.wav')]
    
    print(f"Real: {len(real_files)}, AI: {len(ai_files)}")
    
    # Split data
    real_train, real_test = train_test_split(real_files, test_size=0.2, random_state=42)
    ai_train, ai_test = train_test_split(ai_files, test_size=0.2, random_state=42)
    
    # Use augmented dataset for real-world robustness
    train_dataset = RealWorldAudioDataset(real_train, ai_train, augment_real=True)
    test_dataset = RealWorldAudioDataset(real_test, ai_test, augment_real=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # Focus on correctly identifying real voices
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.5, 1.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.000005)  # Very small LR
    
    print("Training with real-world augmentation...")
    
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output_tuple = model(data)
            output = output_tuple[1]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Evaluate
        model.eval()
        real_correct = 0
        real_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output_tuple = model(data)
                output = output_tuple[1]
                preds = output.argmax(dim=1)
                
                real_mask = target == 0
                if real_mask.sum() > 0:
                    real_correct += (preds[real_mask] == target[real_mask]).sum().item()
                    real_total += real_mask.sum().item()
        
        if real_total > 0:
            real_accuracy = real_correct / real_total
            avg_loss = running_loss / len(train_loader)
            print(f'Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}, Real Acc: {real_accuracy:.4f}')
    
    # Save real-world robust model
    torch.save(model.state_dict(), 'models/aasist_real_world.pth')
    print("Real-world robust model saved!")

if __name__ == "__main__":
    fix_real_world_detection()