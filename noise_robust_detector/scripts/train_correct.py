import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import json
from sklearn.model_selection import train_test_split

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
from models.RawNet2Spoof import Model

# Config
SAMPLE_RATE = 16000
AUDIO_LENGTH = 64600
BATCH_SIZE = 192
EPOCHS = 30
LEARNING_RATE = 0.0001

# PATHS
REAL_PATH = "data/raw/clean_real"
AI_PATH = "data/raw/clean_ai"

class CorrectedDataset(Dataset):
    """CORRECT LABELS: Real=1, AI=0"""
    def __init__(self, real_files, ai_files, max_samples=None):
        self.files = []
        self.labels = []
        
        # REAL voices = label 1
        for f in real_files[:max_samples] if max_samples else real_files:
            self.files.append(f)
            self.labels.append(1)  # REAL = 1
        
        # AI voices = label 0  
        for f in ai_files[:max_samples] if max_samples else ai_files:
            self.files.append(f)
            self.labels.append(0)  # AI = 0
        
        print(f"Dataset: {len(real_files)} real (label 1), {len(ai_files)} AI (label 0)")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio, sr = librosa.load(self.files[idx], sr=SAMPLE_RATE, duration=4.04)
        
        if len(audio) > AUDIO_LENGTH:
            audio = audio[:AUDIO_LENGTH]
        elif len(audio) < AUDIO_LENGTH:
            audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)))
        
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        return torch.FloatTensor(audio), torch.tensor(self.labels[idx], dtype=torch.long)

def train_correct():
    print("=" * 60)
    print("🔄 RETRAINING WITH CORRECT LABELS")
    print("Real = 1, AI = 0")
    print("=" * 60)
    
    # Get ALL files
    real_files = [os.path.join(REAL_PATH, f) for f in os.listdir(REAL_PATH) 
                  if f.endswith('.wav')]
    ai_files = [os.path.join(AI_PATH, f) for f in os.listdir(AI_PATH) 
                if f.endswith('.wav')]
    
    # Split
    real_train, real_temp = train_test_split(real_files, test_size=0.3, random_state=42)
    ai_train, ai_temp = train_test_split(ai_files, test_size=0.3, random_state=42)
    real_val, real_test = train_test_split(real_temp, test_size=0.5, random_state=42)
    ai_val, ai_test = train_test_split(ai_temp, test_size=0.5, random_state=42)
    
    # Create datasets with CORRECT labels
    train_dataset = CorrectedDataset(real_train, ai_train)
    val_dataset = CorrectedDataset(real_val, ai_val)
    test_dataset = CorrectedDataset(real_test, ai_test)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)
    
    # Model
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
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train from scratch - NO pretrained weights
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Training loop
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)[1]  # RawNet2 returns (hidden, logsoftmax)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = output.argmax(dim=1)
            train_correct += (preds == target).sum().item()
            train_total += target.size(0)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)[1]
                preds = output.argmax(dim=1)
                val_correct += (preds == target).sum().item()
                val_total += target.size(0)
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'models/rawnet2_correct.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Final test
    model.load_state_dict(torch.load('models/rawnet2_correct.pth'))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)[1]
            preds = output.argmax(dim=1)
            test_correct += (preds == target).sum().item()
            test_total += target.size(0)
    
    test_acc = 100 * test_correct / test_total
    print(f"\n🎯 FINAL TEST ACCURACY: {test_acc:.1f}%")
    print(f"💾 Model saved: models/rawnet2_correct.pth")
    
    return model

if __name__ == "__main__":
    train_correct()