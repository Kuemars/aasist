import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import json
from torch.cuda.amp import autocast, GradScaler

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
from models.AASIST import Model

# OPTIMIZED CONFIG FOR FULL DATASET
SAMPLE_RATE = 16000
AUDIO_LENGTH = 64600
BATCH_SIZE = 128  # Start with 128, can increase to 256
EPOCHS = 50
LEARNING_RATE = 0.0001
USE_AMP = True  # Mixed precision for 2x speed

# PATHS
REAL_PATH = "data/raw/clean_real"
AI_PATH = "data/raw/clean_ai"

def get_model_config():
    config_path = os.path.join(project_root, "config/AASIST.conf")
    default_config = {
        "nb_samp": 64600,
        "nb_classes": 2,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if "model_config" in config:
                    model_config = config["model_config"]
                    model_config["nb_classes"] = 2
                    model_config["nb_samp"] = 64600
                    return model_config
        except:
            pass
    return default_config

class FullDataset(Dataset):
    def __init__(self, real_files, ai_files):
        self.files = real_files + ai_files
        self.labels = [0]*len(real_files) + [1]*len(ai_files)
        print(f"Full dataset: {len(self.files)} samples ({len(real_files)} real, {len(ai_files)} AI)")
    
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

def train_full():
    print("=" * 60)
    print("🚀 FULL DATASET TRAINING - RTX 4080 SUPER")
    print("=" * 60)
    
    # Get ALL files
    real_files = [os.path.join(REAL_PATH, f) for f in os.listdir(REAL_PATH) 
                  if f.endswith('.wav')]
    ai_files = [os.path.join(AI_PATH, f) for f in os.listdir(AI_PATH) 
                if f.endswith('.wav')]
    
    # Create full dataset
    dataset = FullDataset(real_files, ai_files)
    
    # Split: 70% train, 15% val, 15% test
    from sklearn.model_selection import train_test_split
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    print(f"Training: {len(train_idx)} samples")
    print(f"Validation: {len(val_idx)} samples")
    print(f"Test: {len(test_idx)} samples")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)
    
    # Model
    device = torch.device('cuda')
    model_config = get_model_config()
    model = Model(model_config).to(device)
    
    # Load pre-trained weights
    weights_path = os.path.join(project_root, "models/weights/AASIST.pth")
    if os.path.exists(weights_path):
        print(f"Loading pre-trained weights...")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # Mixed precision
    scaler = GradScaler() if USE_AMP else None
    
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Early stopping
    best_val_acc = 0
    patience_counter = 0
    patience = 10
    
    print(f"\n⚙️ Config: Batch={BATCH_SIZE}, Epochs={EPOCHS}, Mixed Precision={USE_AMP}")
    print("=" * 60)
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            if USE_AMP and scaler:
                with autocast():
                    output = model(data)[1]
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)[1]
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            preds = output.argmax(dim=1)
            train_correct += (preds == target).sum().item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss={loss.item():.4f}")
        
        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)[1]
                preds = output.argmax(dim=1)
                val_correct += (preds == target).sum().item()
        
        train_acc = 100 * train_correct / len(train_idx)
        val_acc = 100 * val_correct / len(val_idx)
        
        print(f"\n📊 Epoch {epoch+1}:")
        print(f"   Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.1f}%")
        print(f"   Val Acc: {val_acc:.1f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'models/aasist_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
    
    # Load best model and test
    model.load_state_dict(torch.load('models/aasist_best.pth'))
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)[1]
            preds = output.argmax(dim=1)
            test_correct += (preds == target).sum().item()
    
    test_acc = 100 * test_correct / len(test_idx)
    print(f"\n🎯 FINAL TEST ACCURACY: {test_acc:.1f}%")
    print(f"💾 Best model saved to models/aasist_best.pth")
    print("=" * 60)

if __name__ == "__main__":
    train_full()