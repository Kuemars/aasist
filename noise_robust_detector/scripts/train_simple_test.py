import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import json  # Changed from yaml to json

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
from models.AASIST import Model

# SIMPLE CONFIG
SAMPLE_RATE = 16000
AUDIO_LENGTH = 64600
BATCH_SIZE = 64  # Can be larger now
EPOCHS = 5

# PATHS
REAL_PATH = "data/raw/clean_real"
AI_PATH = "data/raw/clean_ai"

def get_model_config():
    """Get AASIST config - it's JSON format"""
    config_path = os.path.join(project_root, "config/AASIST.conf")
    
    # Default config matching your file structure
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
                config = json.load(f)  # JSON, not YAML
                if "model_config" in config:
                    model_config = config["model_config"]
                    # Ensure required fields
                    model_config["nb_classes"] = 2
                    model_config["nb_samp"] = 64600
                    print(f"✅ Loaded config from {config_path}")
                    return model_config
        except Exception as e:
            print(f"⚠️ Error loading config: {e}")
    
    print("⚠️ Using default config")
    return default_config

class SimpleDataset(Dataset):
    def __init__(self, real_files, ai_files, max_samples=200):
        self.files = []
        self.labels = []
        
        # Use more samples for better training
        for f in real_files[:max_samples]:
            self.files.append(f)
            self.labels.append(0)
        for f in ai_files[:max_samples]:
            self.files.append(f)
            self.labels.append(1)
        
        print(f"Dataset: {len(self.files)} samples ({max_samples} real, {max_samples} AI)")
    
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

def simple_train():
    print("=" * 60)
    print("🚀 AASIST TRAINING - RTX 4080 SUPER")
    print("=" * 60)
    
    # Get files
    real_files = [os.path.join(REAL_PATH, f) for f in os.listdir(REAL_PATH) 
                  if f.endswith('.wav')]
    ai_files = [os.path.join(AI_PATH, f) for f in os.listdir(AI_PATH) 
                if f.endswith('.wav')]
    
    print(f"Available: {len(real_files)} real, {len(ai_files)} AI")
    
    # Create dataset (use 400 samples total for quick test)
    dataset = SimpleDataset(real_files, ai_files, max_samples=200)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Model
    device = torch.device('cuda')
    print(f"\n📱 Device: {device}")
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    model_config = get_model_config()
    model = Model(model_config).to(device)
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load pre-trained weights
    weights_path = os.path.join(project_root, "models/weights/AASIST.pth")
    if os.path.exists(weights_path):
        print(f"📥 Loading weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("✅ Weights loaded")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    print(f"\n⚙️ Config: Batch={BATCH_SIZE}, Epochs={EPOCHS}, LR=0.0001")
    print("=" * 60)
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)[1]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            
            if batch_idx % 2 == 0:
                batch_acc = 100 * (preds == target).sum().item() / target.size(0)
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss={loss.item():.4f}, Acc={batch_acc:.1f}%")
        
        epoch_acc = 100 * correct / total
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Avg Loss: {total_loss/len(loader):.4f}")
        print(f"   Accuracy: {epoch_acc:.1f}%")
        print("-" * 40)
    
    # Final evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    
    print("\n🧪 Final evaluation...")
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)[1]
            preds = output.argmax(dim=1)
            test_correct += (preds == target).sum().item()
            test_total += target.size(0)
    
    final_acc = 100 * test_correct / test_total
    print(f"\n🎯 FINAL TEST ACCURACY: {final_acc:.1f}%")
    print("=" * 60)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/aasist_trained.pth')
    print(f"💾 Model saved to models/aasist_trained.pth")

if __name__ == "__main__":
    simple_train()