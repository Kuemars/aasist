# scripts/train_ultra_fast.py
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
import time

# ==================== OPTIMIZED CONFIG ====================
SAMPLE_RATE = 16000
AUDIO_LENGTH = 64600
BATCH_SIZE = 128  # HUGE batch for RTX 4080 SUPER (16GB)
EPOCHS = 50
LEARNING_RATE = 0.001  # Higher learning rate for faster convergence
PATIENCE = 10

# ==================== PATHS ====================
REAL_PATH = r"D:\sk13382\DataTraining\aasist\noise_robust_detector\data\raw\clean_real"
AI_PATH = r"D:\sk13382\DataTraining\aasist\noise_robust_detector\data\raw\clean_ai"

# ==================== PRELOAD ALL DATA INTO GPU ====================
def preload_all_audio_gpu(real_files, ai_files, device):
    """Preload ALL audio data into GPU memory for lightning fast training"""
    print("⚡ PRELOADING ALL AUDIO TO GPU MEMORY...")
    
    all_audio = []
    all_labels = []
    
    # Load real voices
    print(f"📥 Loading {len(real_files)} real voices...")
    for i, file in enumerate(real_files):
        audio, _ = librosa.load(file, sr=SAMPLE_RATE, duration=4.04)
        if len(audio) > AUDIO_LENGTH:
            audio = audio[:AUDIO_LENGTH]
        elif len(audio) < AUDIO_LENGTH:
            audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)))
        all_audio.append(audio)
        all_labels.append(0)  # Real = 0
        
        if i % 100 == 0:
            print(f"  Real: {i}/{len(real_files)}")
    
    # Load AI voices
    print(f"📥 Loading {len(ai_files)} AI voices...")
    for i, file in enumerate(ai_files):
        audio, _ = librosa.load(file, sr=SAMPLE_RATE, duration=4.04)
        if len(audio) > AUDIO_LENGTH:
            audio = audio[:AUDIO_LENGTH]
        elif len(audio) < AUDIO_LENGTH:
            audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)))
        all_audio.append(audio)
        all_labels.append(1)  # AI = 1
        
        if i % 100 == 0:
            print(f"  AI: {i}/{len(ai_files)}")
    
    # Convert to tensors and move to GPU
    print("🚀 Moving data to GPU...")
    audio_tensor = torch.FloatTensor(np.array(all_audio)).to(device)
    labels_tensor = torch.LongTensor(np.array(all_labels)).to(device)
    
    print(f"✅ Preloaded {len(all_audio)} samples to GPU memory")
    print(f"   Audio tensor shape: {audio_tensor.shape}")
    print(f"   GPU memory usage: {audio_tensor.element_size() * audio_tensor.nelement() / 1e9:.2f} GB")
    
    return audio_tensor, labels_tensor

# ==================== GPU DATASET (NO DISK I/O) ====================
class GPUDataset(Dataset):
    """Dataset that uses preloaded GPU tensors - ZERO disk I/O during training"""
    def __init__(self, audio_tensor, label_tensor, indices):
        self.audio_tensor = audio_tensor
        self.label_tensor = label_tensor
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.audio_tensor[actual_idx], self.label_tensor[actual_idx]

# ==================== FAST COLLATE ====================
def gpu_collate_fn(batch):
    """Simple collate - data is already on GPU"""
    audio = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return audio, labels

# ==================== TRAINING FUNCTION ====================
def train_ultra_fast():
    print("=" * 70)
    print("⚡ ULTRA-FAST AASIST TRAINING")
    print("=" * 70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎮 Device: {device}")
    
    if torch.cuda.is_available():
        print(f"💾 GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    
    # ==================== LOAD AND PREPROCESS DATA ====================
    print("\n📁 Loading file paths...")
    
    real_files = [os.path.join(REAL_PATH, f) for f in sorted(os.listdir(REAL_PATH)) 
                  if f.startswith('real_') and f.endswith('.wav')][:2000]  # Limit to 2000
    ai_files = [os.path.join(AI_PATH, f) for f in sorted(os.listdir(AI_PATH)) 
                if f.startswith('ai_') and f.endswith('.wav')][:2000]  # Limit to 2000
    
    print(f"✅ Real: {len(real_files)} files")
    print(f"✅ AI: {len(ai_files)} files")
    
    # Preload ALL data to GPU
    audio_tensor, labels_tensor = preload_all_audio_gpu(real_files, ai_files, device)
    
    # ==================== CREATE TRAIN/VAL/TEST SPLITS ====================
    print("\n📊 Creating splits...")
    
    # Create indices for splitting
    total_samples = len(audio_tensor)
    indices = np.arange(total_samples)
    
    # Split indices
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels_tensor.cpu().numpy())
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, 
                                         stratify=labels_tensor[temp_idx].cpu().numpy())
    
    print(f"   Training:   {len(train_idx)} samples")
    print(f"   Validation: {len(val_idx)} samples")
    print(f"   Test:       {len(test_idx)} samples")
    
    # Create GPU datasets
    train_dataset = GPUDataset(audio_tensor, labels_tensor, train_idx)
    val_dataset = GPUDataset(audio_tensor, labels_tensor, val_idx)
    test_dataset = GPUDataset(audio_tensor, labels_tensor, test_idx)
    
    # Create dataloaders (no workers needed since data is on GPU)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=gpu_collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=gpu_collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=gpu_collate_fn, num_workers=0)
    
    # ==================== LOAD MODEL ====================
    print("\n🤖 Loading model...")
    
    # Import AASIST
    script_dir = os.path.dirname(os.path.abspath(__file__))
    aasist_root = os.path.dirname(os.path.dirname(script_dir))
    sys.path.insert(0, aasist_root)
    
    from models.AASIST import Model
    
    # Get config
    config_path = os.path.join(aasist_root, "config", "AASIST.conf")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        model_config = config.get('model_config', config)
    else:
        model_config = {
            'architecture': 'AASIST',
            'nb_samp': 64600,
            'first_conv': 128,
            'filts': [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            'gat_dims': [64, 32],
            'pool_ratios': [0.5, 0.7, 0.5, 0.5],
            'temperatures': [2.0, 2.0, 100.0, 100.0],
            'nb_classes': 2,
        }
    
    # Create model
    model = Model(model_config).to(device)
    
    # Load pretrained weights if available
    weights_path = os.path.join(aasist_root, "models", "weights", "AASIST.pth")
    if os.path.exists(weights_path):
        print(f"📥 Loading pretrained weights...")
        try:
            checkpoint = torch.load(weights_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("✅ Weights loaded")
        except:
            print("⚠️  Could not load weights, starting fresh")
    
    # ==================== SETUP TRAINING ====================
    print("\n⚙️  Setting up training...")
    
    # Mixed precision training for speed
    scaler = torch.cuda.amp.GradScaler()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Learning rate: {LEARNING_RATE}")
    
    # ==================== TRAINING LOOP ====================
    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70)
    
    start_time = time.time()
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # ========== TRAINING ==========
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Data is already on GPU from preloading
            
            optimizer.zero_grad()
            
            # Mixed precision forward
            with torch.cuda.amp.autocast():
                output = model(data)
                if isinstance(output, tuple):
                    output = output[1]  # Get classification output
                loss = criterion(output, target)
            
            # Mixed precision backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # Fast progress indicator
            if batch_idx % 5 == 0:
                batch_acc = 100. * predicted.eq(target).sum().item() / target.size(0)
                print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}, Acc: {batch_acc:.1f}%", end='\r')
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # ========== VALIDATION ==========
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    if isinstance(output, tuple):
                        output = output[1]
                
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_acc = 100. * val_correct / val_total
        epoch_time = time.time() - epoch_start
        
        # ========== EPOCH SUMMARY ==========
        print(f"\n📊 Epoch {epoch+1}/{EPOCHS} - {epoch_time:.1f}s")
        print(f"   Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"   Val Acc: {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'config': model_config
            }, 'best_model.pth')
            
            print(f"   💾 Saved best model: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
        
        print("-" * 50)
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time/60:.1f} minutes")
    
    # ==================== FINAL EVALUATION ====================
    print("\n🧪 Final evaluation...")
    
    # Load best model
    if os.path.exists('best_model.pth'):
        checkpoint = torch.load('best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"📥 Loaded best model from epoch {checkpoint['epoch'] + 1}")
    
    # Test evaluation
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            with torch.cuda.amp.autocast():
                output = model(data)
                if isinstance(output, tuple):
                    output = output[1]
            
            _, predicted = output.max(1)
            test_preds.extend(predicted.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
    
    test_acc = accuracy_score(test_targets, test_preds) * 100
    
    print(f"\n🎯 FINAL TEST ACCURACY: {test_acc:.2f}%")
    print("\n📈 Classification Report:")
    print(classification_report(test_targets, test_preds, target_names=['Real', 'AI']))
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_accuracy': test_acc,
        'config': model_config
    }, 'final_model.pth')
    
    print(f"\n💾 Models saved:")
    print(f"   - best_model.pth (best validation: {best_val_acc:.2f}%)")
    print(f"   - final_model.pth (test: {test_acc:.2f}%)")
    
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)

# ==================== MAIN ====================
if __name__ == "__main__":
    train_ultra_fast()