import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import yaml

# Add paths for AASIST import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.AASIST import Model

# Configuration
SAMPLE_RATE = 16000
AUDIO_LENGTH = 64600
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.0001

# Paths
REAL_PATH = "data/raw/clean_real"
AI_PATH = "data/raw/clean_ai"
PRETRAINED_WEIGHTS = "../models/weights/AASIST.pth"

class AudioDataset(Dataset):
    def __init__(self, real_files, ai_files):
        self.files = []
        self.labels = []
        
        # Add real voices (label 0)
        for file in real_files:
            self.files.append(file)
            self.labels.append(0)
        
        # Add AI voices (label 1)
        for file in ai_files:
            self.files.append(file)
            self.labels.append(1)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        
        # Load audio - EXACTLY like create_model.py
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

        # Ensure correct length - EXACTLY like create_model.py
        if len(audio) > AUDIO_LENGTH:
            audio = audio[:AUDIO_LENGTH]
        else:
            padding = AUDIO_LENGTH - len(audio)
            audio = np.pad(audio, (0, padding))

        # Convert to tensor - EXACTLY like create_model.py: [64600] not [1, 64600]
        audio_tensor = torch.FloatTensor(audio)  # This is [64600]
        label_tensor = torch.LongTensor([self.labels[idx]])
        
        return audio_tensor, label_tensor

def collate_fn(batch):
    """Custom collate function to match create_model.py input format"""
    audio_tensors = []
    labels = []
    
    for audio, label in batch:
        # Keep as [64600] - will stack to [batch_size, 64600]
        audio_tensors.append(audio)
        labels.append(label)
    
    # Stack to [batch_size, 64600] - this is what AASIST expects
    audio_batch = torch.stack(audio_tensors)  # Shape: [batch_size, 64600]
    label_batch = torch.stack(labels).squeeze()
    
    return audio_batch, label_batch

def get_aasist_config():
    """Load the exact config from file like create_model.py"""
    config_path = "../config/AASIST.conf"
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config['model_config']

def fine_tune_aasist():
    print("Starting AASIST fine-tuning...")
    
    # Get all audio files
    real_files = [os.path.join(REAL_PATH, f) for f in os.listdir(REAL_PATH) if f.endswith('.wav')]
    ai_files = [os.path.join(AI_PATH, f) for f in os.listdir(AI_PATH) if f.endswith('.wav')]
    
    print(f"Real voices: {len(real_files)}")
    print(f"AI voices: {len(ai_files)}")
    
    # Split data
    real_train, real_test = train_test_split(real_files, test_size=0.2, random_state=42)
    ai_train, ai_test = train_test_split(ai_files, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = AudioDataset(real_train, ai_train)
    test_dataset = AudioDataset(real_test, ai_test)
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Initialize AASIST model - EXACTLY like create_model.py
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config from file like create_model.py
    model_config = get_aasist_config()
    model = Model(model_config).to(device)
    
    # Load pre-trained weights - EXACTLY like create_model.py
    if os.path.exists(PRETRAINED_WEIGHTS):
        print(f"Loading pre-trained weights from {PRETRAINED_WEIGHTS}")
        state_dict = torch.load(PRETRAINED_WEIGHTS, map_location=device)
        model.load_state_dict(state_dict)
        print("Pre-trained weights loaded successfully")
    else:
        print("Warning: Pre-trained weights not found. Training from scratch.")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Model returns tuple, we need the second element like create_model.py
            output_tuple = model(data)
            output = output_tuple[1]  # This is the [batch_size, 2] tensor
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Evaluate
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output_tuple = model(data)
                output = output_tuple[1]  # Get the [batch_size, 2] tensor
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = accuracy_score(all_targets, all_preds)
        avg_loss = running_loss / len(train_loader)
        
        print(f'Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')
    
    # Save fine-tuned model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/aasist_fine_tuned.pth')
    print("Fine-tuned model saved to models/aasist_fine_tuned.pth")
    
    # Final evaluation
    print("\nFinal Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=['Real', 'AI']))

if __name__ == "__main__":
    fine_tune_aasist()