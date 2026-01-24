import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import librosa
import numpy as np
import json
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
import time

# ============================================================================
# FIXED PATH CONFIGURATION
# ============================================================================

# Get the absolute path of THIS script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Determine if we're running from scripts/ folder or root
if os.path.basename(current_script_dir) == 'scripts':
    # Running from scripts folder
    PROJECT_ROOT = os.path.dirname(current_script_dir)  # Go up one level
    print(f"📂 Running from scripts folder, root: {PROJECT_ROOT}")
else:
    # Running from root or elsewhere
    PROJECT_ROOT = current_script_dir
    print(f"📂 Running from root folder: {PROJECT_ROOT}")

# Add project root to Python path
sys.path.insert(0, PROJECT_ROOT)

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

SAMPLE_RATE = 16000
AUDIO_LENGTH = 64600
BATCH_SIZE = 192
EPOCHS = 50
LEARNING_RATE = 0.0001
USE_AMP = True

# ============================================================================
# FIXED PATHS
# ============================================================================

REAL_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "clean_real")
AI_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "clean_ai")

# Model save path
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "weights")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def get_model_config():
    """Load RawNet2 config from file or use default"""
    config_path = os.path.join(PROJECT_ROOT, "config", "RawNet2_baseline.conf")
    default_config = {
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

def preload_all_audio_files(real_files, ai_files):
    """Load all data to CPU RAM first, then move to GPU in batches"""
    print("⚡ Loading ALL data to RAM...")
    
    all_audio = []
    all_labels = []
    
    # Load real voices
    print(f"📂 Loading real voices from: {REAL_PATH}")
    for i, file in enumerate(real_files):
        if i % 400 == 0:
            print(f"  Real: {i}/{len(real_files)}")
        audio, _ = librosa.load(file, sr=SAMPLE_RATE, duration=4.04)
        if len(audio) > AUDIO_LENGTH:
            audio = audio[:AUDIO_LENGTH]
        else:
            audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)))
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        all_audio.append(audio)
        all_labels.append(0)
    
    # Load AI voices
    print(f"📂 Loading AI voices from: {AI_PATH}")
    for i, file in enumerate(ai_files):
        if i % 400 == 0:
            print(f"  AI: {i}/{len(ai_files)}")
        audio, _ = librosa.load(file, sr=SAMPLE_RATE, duration=4.04)
        if len(audio) > AUDIO_LENGTH:
            audio = audio[:AUDIO_LENGTH]
        else:
            audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)))
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        all_audio.append(audio)
        all_labels.append(1)
    
    print(f"✅ Loaded {len(all_audio)} samples to RAM")
    return np.array(all_audio, dtype=np.float32), np.array(all_labels, dtype=np.int64)

def train_optimal():
    print("=" * 60)
    print("⚡ OPTIMAL TRAINING - BATCH 192 + PRE-LOADED")
    print("=" * 60)
    
    start_time = time.time()
    
    # Get ALL files
    real_files = [os.path.join(REAL_PATH, f) for f in os.listdir(REAL_PATH) 
                  if f.endswith('.wav')]
    ai_files = [os.path.join(AI_PATH, f) for f in os.listdir(AI_PATH) 
                if f.endswith('.wav')]
    
    print(f"📊 Dataset: {len(real_files)} real, {len(ai_files)} AI")
    
    # PRE-LOAD to RAM (not GPU yet)
    audio_data, labels_data = preload_all_audio_files(real_files, ai_files)
    load_time = time.time() - start_time
    print(f"⏱️ Data loading time: {load_time:.1f}s")
    
    # Split indices
    indices = np.arange(len(audio_data))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, shuffle=True)
    
    print(f"📈 Splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Device
    device = torch.device('cuda')
    print(f"🖥️ Device: {device}")
    print(f"💾 GPU Memory before: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Create datasets (will load to GPU on-demand)
    train_dataset = TensorDataset(
        torch.FloatTensor(audio_data[train_idx]), 
        torch.LongTensor(labels_data[train_idx])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(audio_data[val_idx]), 
        torch.LongTensor(labels_data[val_idx])
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(audio_data[test_idx]), 
        torch.LongTensor(labels_data[test_idx])
    )
    
    # DataLoaders with pin_memory for fast GPU transfer
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE*2, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE*2, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    # Model - FIXED IMPORT PATH
    try:
        # Try relative import
        from models.RawNet2Spoof import Model
        print("✅ Successfully imported RawNet2Spoof model")
    except ImportError:
        # Try absolute import
        import sys
        sys.path.append(PROJECT_ROOT)
        from models.RawNet2Spoof import Model
        print("✅ Successfully imported RawNet2Spoof model (using absolute path)")
    
    model_config = get_model_config()
    model = Model(model_config).to(device)
    
    # Mixed precision
    scaler = GradScaler() if USE_AMP else None
    
    # Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Early stopping
    best_val_acc = 0
    patience_counter = 0
    patience = 10
    
    print(f"\n⚙️ Config: Batch={BATCH_SIZE}, Epochs={EPOCHS}, Mixed Precision={USE_AMP}")
    print("=" * 60)
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
            preds = output.argmax(dim=1)
            train_correct += (preds == target).sum().item()
            batch_count += 1
            
            # Show progress every 2 batches
            if batch_idx % 2 == 0:
                batch_acc = 100 * (preds == target).sum().item() / target.size(0)
                print(f"📊 Epoch {epoch+1}, Batch {batch_idx}: Loss={loss.item():.4f}, Acc={batch_acc:.1f}%")
        
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
        epoch_time = time.time() - epoch_start
        
        print(f"\n📈 Epoch {epoch+1} ({epoch_time:.1f}s):")
        print(f"   Train Loss: {train_loss/batch_count:.4f}, Train Acc: {train_acc:.1f}%")
        print(f"   Val Acc: {val_acc:.1f}%")
        print(f"   Batches: {batch_count}, Time/batch: {epoch_time/batch_count:.2f}s")
        
        # Early stopping & save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model_save_path = os.path.join(MODEL_SAVE_DIR, 'rawnet2_v1.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"   🏆 New best! Saving model to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
    
    # Final test with best model
    best_model_path = os.path.join(MODEL_SAVE_DIR, 'rawnet2_v1.pth')
    if os.path.exists(best_model_path):
        print(f"🔍 Loading best model from {best_model_path} for testing...")
        model.load_state_dict(torch.load(best_model_path))
    
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)[1]
            preds = output.argmax(dim=1)
            test_correct += (preds == target).sum().item()
    
    test_acc = 100 * test_correct / len(test_idx)
    total_time = time.time() - start_time
    
    # SAVE RESULTS
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "training_results.json")
    
    results = {
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
        "total_training_time_seconds": total_time,
        "total_training_time_minutes": total_time / 60,
        "model_saved_at": best_model_path,
        "training_config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "audio_length": AUDIO_LENGTH,
            "sample_rate": SAMPLE_RATE,
            "use_amp": USE_AMP
        },
        "dataset_stats": {
            "total_samples": len(audio_data),
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "test_samples": len(test_idx),
            "real_files": len(real_files),
            "ai_files": len(ai_files)
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Test Accuracy: {test_acc:.1f}%")
    print(f"   Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"   Total training time: {total_time/60:.1f} minutes")
    print(f"   📄 Results saved to: {results_path}")
    print(f"   💾 Model saved to: {best_model_path}")
    print("=" * 60)

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Run training
    train_optimal()