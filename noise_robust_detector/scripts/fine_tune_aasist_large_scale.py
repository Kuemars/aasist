# scripts/fine_tune_aasist_high_performance.py
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import yaml
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.cuda.amp as amp  # Mixed precision training
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

# Add paths for AASIST import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from models.AASIST import Model
    print("✅ AASIST model imported successfully")
except ImportError as e:
    print(f"❌ Error importing AASIST: {e}")
    sys.exit(1)

# ==================== OPTIMIZED CONFIGURATION ====================
# Tuned for RTX 4090/3090 with 24GB VRAM
SAMPLE_RATE = 16000
AUDIO_LENGTH = 64600  # 4.04 seconds
BATCH_SIZE = 32       # Increased for GPU (was 8)
EPOCHS = 50           # More epochs for large dataset
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5   # L2 regularization
PATIENCE = 10         # Early stopping patience
GRADIENT_ACCUMULATION_STEPS = 1  # For even larger batches if needed
MIXED_PRECISION = True  # Use AMP for faster training

# ==================== PATHS ====================
REAL_PATH = r"D:\sk13382\DataTraining\aasist\noise_robust_detector\data\raw\clean_real"
AI_PATH = r"D:\sk13382\DataTraining\aasist\noise_robust_detector\data\raw\clean_ai"
PRETRAINED_WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                  "models", "weights", "AASIST.pth")

# ==================== OPTIMIZED DATASET ====================
class HighPerformanceAudioDataset(Dataset):
    """Optimized dataset with caching for faster training"""
    
    def __init__(self, real_files, ai_files, cache_dir=None):
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
        
        self.cache_dir = cache_dir
        self.cache = {}
        
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        print(f"📊 Dataset: {len(real_files)} real + {len(ai_files)} AI = {len(self.files)} total")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Check cache first
        cache_key = f"{self.labels[idx]}_{os.path.basename(self.files[idx])}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        audio_path = self.files[idx]
        
        try:
            # Load audio with optimized settings
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            
            # Ensure correct length with minimal operations
            if len(audio) != AUDIO_LENGTH:
                if len(audio) > AUDIO_LENGTH:
                    start = (len(audio) - AUDIO_LENGTH) // 2
                    audio = audio[start:start + AUDIO_LENGTH]
                else:
                    audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)))
            
            # Normalize audio (helps convergence)
            audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)
            
            audio_tensor = torch.FloatTensor(audio)
            label_tensor = torch.LongTensor([self.labels[idx]])
            
            result = (audio_tensor, label_tensor)
            
            # Cache if enabled
            if self.cache_dir:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"❌ Error loading {audio_path}: {e}")
            # Return zeros as fallback
            return torch.zeros(AUDIO_LENGTH), torch.LongTensor([self.labels[idx]])

# ==================== OPTIMIZED COLLATE FUNCTION ====================
def optimized_collate_fn(batch):
    """Optimized collate function with pre-allocation"""
    batch_size = len(batch)
    
    # Pre-allocate tensors
    audio_batch = torch.zeros((batch_size, AUDIO_LENGTH), dtype=torch.float32)
    label_batch = torch.zeros((batch_size,), dtype=torch.long)
    
    for i, (audio, label) in enumerate(batch):
        audio_batch[i] = audio
        label_batch[i] = label.squeeze()
    
    return audio_batch, label_batch

# ==================== OPTIMIZED MODEL TRAINING ====================
class HighPerformanceTrainer:
    def __init__(self, model, device, mixed_precision=True):
        self.model = model
        self.device = device
        self.mixed_precision = mixed_precision
        self.scaler = amp.GradScaler() if mixed_precision else None
        
    def train_epoch(self, train_loader, criterion, optimizer, epoch):
        """Optimized training epoch with mixed precision"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            if self.mixed_precision:
                # Mixed precision forward pass
                with amp.autocast():
                    output_tuple = self.model(data)
                    output = output_tuple[1]
                    loss = criterion(output, target) / GRADIENT_ACCUMULATION_STEPS
                
                # Scaled backward pass
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                # Standard forward/backward
                output_tuple = self.model(data)
                output = output_tuple[1]
                loss = criterion(output, target) / GRADIENT_ACCUMULATION_STEPS
                loss.backward()
                
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Statistics
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item()*GRADIENT_ACCUMULATION_STEPS:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader, criterion):
        """Optimized evaluation"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='[Val]', leave=False)
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.mixed_precision:
                    with amp.autocast():
                        output_tuple = self.model(data)
                        output = output_tuple[1]
                        loss = criterion(output, target)
                else:
                    output_tuple = self.model(data)
                    output = output_tuple[1]
                    loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                pbar.set_postfix({
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, all_preds, all_targets

# ==================== MAIN TRAINING FUNCTION ====================
def high_performance_fine_tune():
    print("=" * 70)
    print("🚀 AASIST HIGH-PERFORMANCE FINE-TUNING")
    print("=" * 70)
    
    # Check CUDA availability and capabilities
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.2f} GB")
        print(f"🎯 CUDA Version: {torch.version.cuda}")
        
        # Enable benchmarking for optimal performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        print("⚠️  No GPU detected! Training will be very slow.")
        MIXED_PRECISION = False
    
    # ==================== LOAD DATASET ====================
    print("\n📁 Loading dataset...")
    
    real_files = [os.path.join(REAL_PATH, f) for f in sorted(os.listdir(REAL_PATH)) 
                  if f.startswith('real_') and f.endswith('.wav')]
    ai_files = [os.path.join(AI_PATH, f) for f in sorted(os.listdir(AI_PATH)) 
                if f.startswith('ai_') and f.endswith('.wav')]
    
    print(f"✅ Real voices: {len(real_files)}")
    print(f"✅ AI voices: {len(ai_files)}")
    
    if len(real_files) == 0 or len(ai_files) == 0:
        print("❌ No files found!")
        return
    
    # Balance dataset
    min_samples = min(len(real_files), len(ai_files))
    real_files = real_files[:min_samples]
    ai_files = ai_files[:min_samples]
    
    print(f"📊 Balanced: {len(real_files)} real + {len(ai_files)} AI = {len(real_files) * 2} total")
    
    # ==================== SPLIT DATASET ====================
    print("\n📈 Creating splits (70/15/15)...")
    
    # Real files split
    real_train, real_temp = train_test_split(real_files, test_size=0.3, random_state=42, shuffle=True)
    real_val, real_test = train_test_split(real_temp, test_size=0.5, random_state=42, shuffle=True)
    
    # AI files split
    ai_train, ai_temp = train_test_split(ai_files, test_size=0.3, random_state=42, shuffle=True)
    ai_val, ai_test = train_test_split(ai_temp, test_size=0.5, random_state=42, shuffle=True)
    
    print(f"   Training:   {len(real_train)} real + {len(ai_train)} AI = {len(real_train) + len(ai_train)}")
    print(f"   Validation: {len(real_val)} real + {len(ai_val)} AI = {len(real_val) + len(ai_val)}")
    print(f"   Test:       {len(real_test)} real + {len(ai_test)} AI = {len(real_test) + len(ai_test)}")
    
    # ==================== CREATE DATALOADERS ====================
    print("\n🔄 Creating dataloaders...")
    
    # Cache training data for speed
    cache_dir = "data_cache" if os.path.exists("data_cache") else None
    
    train_dataset = HighPerformanceAudioDataset(real_train, ai_train, cache_dir)
    val_dataset = HighPerformanceAudioDataset(real_val, ai_val)
    test_dataset = HighPerformanceAudioDataset(real_test, ai_test)
    
    # Use multiple workers for data loading
    num_workers = min(4, os.cpu_count() // 2)
    print(f"   Data workers: {num_workers}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=num_workers, pin_memory=True,
                             collate_fn=optimized_collate_fn, persistent_workers=True)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=num_workers, pin_memory=True,
                           collate_fn=optimized_collate_fn, persistent_workers=True)
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            collate_fn=optimized_collate_fn, persistent_workers=True)
    
    # ==================== LOAD MODEL ====================
    print("\n🤖 Loading model...")
    
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              "config", "AASIST.conf")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model_config = config['model_config']
    model_config['nb_classes'] = 2  # Ensure binary classification
    
    model = Model(model_config).to(device)
    
    # Load pretrained weights if available
    if os.path.exists(PRETRAINED_WEIGHTS):
        print(f"📥 Loading pretrained weights: {PRETRAINED_WEIGHTS}")
        try:
            state_dict = torch.load(PRETRAINED_WEIGHTS, map_location=device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            elif 'state_dict' in state_dict:
                model.load_state_dict(state_dict['state_dict'])
            else:
                model.load_state_dict(state_dict)
            
            print("✅ Pretrained weights loaded")
        except Exception as e:
            print(f"⚠️  Error loading weights: {e}")
            print("   Starting from scratch")
    else:
        print("⚠️  Pretrained weights not found. Starting from scratch.")
    
    # ==================== SETUP TRAINING ====================
    print("\n⚙️  Setting up training...")
    
    # Loss function with class weights (optional)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, 
                          weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=LEARNING_RATE/100)
    
    # Create trainer
    trainer = HighPerformanceTrainer(model, device, mixed_precision=MIXED_PRECISION)
    
    # ==================== TRAINING LOOP ====================
    print("\n🚀 Starting training...")
    print("=" * 70)
    
    start_time = time.time()
    best_val_acc = 0
    patience_counter = 0
    
    # Training history
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader, criterion, optimizer, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = trainer.evaluate(val_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(epoch + val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1}/{EPOCHS} ({epoch_time:.1f}s):")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_acc': val_acc,
                'config': model_config
            }
            
            torch.save(checkpoint, 'best_model.pth')
            print(f"   💾 Saved new best model: {val_acc:.2f}%")
            
            # Also save ONNX for inference
            if epoch > 0:  # Wait for some training
                try:
                    dummy_input = torch.randn(1, AUDIO_LENGTH).to(device)
                    torch.onnx.export(model, dummy_input, "best_model.onnx",
                                    input_names=['audio'], output_names=['output'],
                                    dynamic_axes={'audio': {0: 'batch_size'}})
                except:
                    pass
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
        
        # Clear cache to save memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time/60:.1f} minutes")
    
    # ==================== FINAL EVALUATION ====================
    print("\n🧪 Final evaluation on test set...")
    
    # Load best model
    if os.path.exists('best_model.pth'):
        checkpoint = torch.load('best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"📥 Loaded best model from epoch {checkpoint['epoch'] + 1}")
    
    # Evaluate on test set
    test_loss, test_acc, test_preds, test_targets = trainer.evaluate(test_loader, criterion)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Test Accuracy:  {test_acc:.2f}%")
    print(f"   Test F1 Score:  {f1_score(test_targets, test_preds, average='macro'):.4f}")
    
    # Detailed metrics
    print("\n📈 Classification Report:")
    print(classification_report(test_targets, test_preds, target_names=['Real', 'AI']))
    
    # Confusion matrix
    cm = confusion_matrix(test_targets, test_preds)
    print("📊 Confusion Matrix:")
    print(cm)
    
    # ==================== SAVE RESULTS ====================
    print("\n💾 Saving results...")
    
    # Create results directory
    results_dir = "training_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'test_predictions': test_preds,
        'test_targets': test_targets
    }
    
    np.save(os.path.join(results_dir, 'training_history.npy'), history)
    
    # Save final model
    final_checkpoint = {
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': test_acc,
        'config': model_config,
        'training_history': history
    }
    
    torch.save(final_checkpoint, os.path.join(results_dir, 'final_model.pth'))
    
    # Save summary
    with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
        f.write(f"AASIST High-Performance Training Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Dataset: {len(real_files)} real + {len(ai_files)} AI\n")
        f.write(f"Training time: {total_time/60:.1f} minutes\n")
        f.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Test accuracy: {test_acc:.2f}%\n")
        f.write(f"Test F1 Score: {f1_score(test_targets, test_preds, average='macro'):.4f}\n")
        f.write(f"Device: {device}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    print(f"\n📊 Results saved to: {results_dir}/")
    print("=" * 70)
    print("🎉 TRAINING COMPLETE! Model is ready for inference.")
    print("=" * 70)
    
    # Display GPU memory usage
    if torch.cuda.is_available():
        print(f"\n💾 GPU Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"   Cached:    {torch.cuda.memory_reserved()/1e9:.2f} GB")

# ==================== RUN TRAINING ====================
if __name__ == "__main__":
    high_performance_fine_tune()