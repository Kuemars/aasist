# scripts/train_precise_fast.py
import os
import sys
sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import yaml
import time

print("🎯 PRECISE & FAST AASIST TRAINING")
print("=" * 70)
print("Based on your working script, optimized for RTX 4080 SUPER")
print("=" * 70)

# ==================== OPTIMAL CONFIGURATION ====================
SAMPLE_RATE = 16000
AUDIO_LENGTH = 64600
BATCH_SIZE = 128  # 2x your original (8 → 128) - RTX 4080 can handle it
EPOCHS = 50       # More epochs for large dataset
LEARNING_RATE = 0.0001
PATIENCE = 8      # Early stopping patience

# ==================== PATHS ====================
REAL_PATH = r"D:\sk13382\DataTraining\aasist\noise_robust_detector\data\raw\clean_real"
AI_PATH = r"D:\sk13382\DataTraining\aasist\noise_robust_detector\data\raw\clean_ai"

# Import AASIST - using your proven method
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from models.AASIST import Model
    print("✅ AASIST model imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# ==================== OPTIMIZED DATASET ====================
class FastAudioDataset(Dataset):
    """
    Optimized version of your working dataset class
    Key optimizations:
    1. Loads all data to GPU at startup
    2. Minimal operations in __getitem__
    3. Pre-normalized audio
    """
    
    def __init__(self, file_paths, labels, device):
        """
        file_paths: list of file paths
        labels: list of labels (0 for real, 1 for AI)
        device: torch device to preload data to
        """
        print(f"🔄 Preloading {len(file_paths)} samples to GPU...")
        
        self.audio_tensors = []
        self.labels_tensor = torch.LongTensor(labels).to(device)
        
        # Preload all audio to GPU
        for i, file_path in enumerate(file_paths):
            try:
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # Ensure correct length (same as your working script)
                if len(audio) > AUDIO_LENGTH:
                    audio = audio[:AUDIO_LENGTH]
                else:
                    padding = AUDIO_LENGTH - len(audio)
                    audio = np.pad(audio, (0, padding))
                
                # Normalize (helps training)
                audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-8)
                
                # Store as tensor on GPU
                audio_tensor = torch.FloatTensor(audio).to(device)
                self.audio_tensors.append(audio_tensor)
                
                # Progress tracking
                if i % 500 == 0:
                    print(f"  Loaded {i}/{len(file_paths)} samples", end='\r')
                    
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                # Add zero tensor as fallback
                self.audio_tensors.append(torch.zeros(AUDIO_LENGTH, device=device))
        
        print(f"✅ Preloaded {len(self.audio_tensors)} samples to GPU")
    
    def __len__(self):
        return len(self.audio_tensors)
    
    def __getitem__(self, idx):
        """Ultra-fast: just return preloaded tensors"""
        return self.audio_tensors[idx], self.labels_tensor[idx]

# ==================== OPTIMIZED COLLATE FUNCTION ====================
def fast_collate_fn(batch):
    """
    Optimized collate function
    Data is already on GPU, just stack it
    """
    audio_tensors = []
    labels = []
    
    for audio, label in batch:
        audio_tensors.append(audio)
        labels.append(label)
    
    # Stack to [batch_size, 64600]
    audio_batch = torch.stack(audio_tensors)
    label_batch = torch.stack(labels)
    
    return audio_batch, label_batch

# ==================== GET MODEL CONFIG ====================
def get_aasist_config():
    """Load the exact config from file like your working script"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              "config", "AASIST.conf")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config['model_config']

# ==================== PRECISE PROGRESS TRACKING ====================
class ProgressTracker:
    """Precise progress tracking without external dependencies"""
    
    def __init__(self, total, description=""):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.last_print = 0
        
    def update(self, n=1):
        self.current += n
        current_time = time.time()
        
        # Update every 2 seconds or when complete
        if current_time - self.last_print > 2 or self.current >= self.total:
            elapsed = current_time - self.start_time
            percent = (self.current / self.total) * 100
            
            # Calculate ETA
            if self.current > 0:
                eta = (elapsed / self.current) * (self.total - self.current)
                eta_str = f"ETA: {eta:.0f}s"
            else:
                eta_str = "ETA: --"
            
            # Progress bar (50 characters)
            bar_length = 50
            filled = int(bar_length * self.current / self.total)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            print(f"\r{self.description} [{bar}] {percent:5.1f}% | {self.current}/{self.total} | {eta_str}", 
                  end="", flush=True)
            
            self.last_print = current_time
            
            if self.current >= self.total:
                print()  # New line when complete

# ==================== MAIN TRAINING FUNCTION ====================
def train_precise_fast():
    print("\n" + "=" * 70)
    print("📊 SYSTEM CHECK")
    print("=" * 70)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.2f} GB")
        
        # Enable GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"⚡ GPU optimizations: Enabled")
    else:
        print("⚠️  No GPU detected - training will be slow!")
        return
    
    print("\n" + "=" * 70)
    print("📁 LOADING DATASET")
    print("=" * 70)
    
    # Get all file paths
    real_files = [os.path.join(REAL_PATH, f) for f in sorted(os.listdir(REAL_PATH)) 
                  if f.startswith('real_') and f.endswith('.wav')]
    ai_files = [os.path.join(AI_PATH, f) for f in sorted(os.listdir(AI_PATH)) 
                if f.startswith('ai_') and f.endswith('.wav')]
    
    print(f"✅ Found: {len(real_files)} real voice files")
    print(f"✅ Found: {len(ai_files)} AI voice files")
    
    if len(real_files) == 0 or len(ai_files) == 0:
        print("❌ No files found in dataset directories!")
        return
    
    # Balance dataset
    min_samples = min(len(real_files), len(ai_files))
    if len(real_files) != len(ai_files):
        print(f"⚠️  Balancing dataset: using {min_samples} samples from each class")
        real_files = real_files[:min_samples]
        ai_files = ai_files[:min_samples]
    
    print(f"📊 Balanced dataset: {len(real_files)} real + {len(ai_files)} AI = {len(real_files) * 2} total")
    
    # Create labels
    real_labels = [0] * len(real_files)  # Real = 0
    ai_labels = [1] * len(ai_files)      # AI = 1
    
    # Combine files and labels
    all_files = real_files + ai_files
    all_labels = real_labels + ai_labels
    
    # Split data (70% train, 15% validation, 15% test)
    print("\n📈 Creating dataset splits...")
    
    # First split: train vs temp (val+test)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    
    # Second split: validation vs test
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"   Training:   {len(train_files)} samples")
    print(f"   Validation: {len(val_files)} samples")
    print(f"   Test:       {len(test_files)} samples")
    
    # ==================== PRELOAD DATASETS TO GPU ====================
    print("\n" + "=" * 70)
    print("⚡ PRELOADING DATA TO GPU")
    print("=" * 70)
    
    # Preload training data
    train_dataset = FastAudioDataset(train_files, train_labels, device)
    
    # Preload validation data
    val_dataset = FastAudioDataset(val_files, val_labels, device)
    
    # Preload test data
    test_dataset = FastAudioDataset(test_files, test_labels, device)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, collate_fn=fast_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                          shuffle=False, collate_fn=fast_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, collate_fn=fast_collate_fn)
    
    print(f"\n✅ Data loaders created:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # ==================== SETUP MODEL ====================
    print("\n" + "=" * 70)
    print("🤖 SETTING UP MODEL")
    print("=" * 70)
    
    # Load config
    model_config = get_aasist_config()
    model = Model(model_config).to(device)
    
    # Load pre-trained weights if available
    pretrained_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                  "models", "weights", "AASIST.pth")
    
    if os.path.exists(pretrained_path):
        print(f"📥 Loading pre-trained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ Pre-trained weights loaded")
    else:
        print("⚠️  No pre-trained weights found. Training from scratch.")
    
    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5, verbose=True
    )
    
    # Mixed precision scaler for faster training
    scaler = torch.cuda.amp.GradScaler()
    
    print(f"\n⚙️  Training configuration:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Total epochs: {EPOCHS}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Early stopping patience: {PATIENCE}")
    print(f"   Mixed precision: Enabled")
    
    # ==================== TRAINING LOOP ====================
    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70)
    
    start_time = time.time()
    best_val_accuracy = 0.0
    patience_counter = 0
    
    # Track metrics
    train_loss_history = []
    val_accuracy_history = []
    
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        
        # ========== TRAINING PHASE ==========
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\n📈 Epoch {epoch+1}/{EPOCHS}")
        print("-" * 50)
        
        # Training progress tracker
        train_progress = ProgressTracker(len(train_loader), "  Training")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                output_tuple = model(data)
                output = output_tuple[1]  # Get classification output
                loss = criterion(output, target)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress
            train_progress.update()
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        train_loss_history.append(avg_train_loss)
        
        # ========== VALIDATION PHASE ==========
        model.eval()
        val_correct = 0
        val_total = 0
        
        val_progress = ProgressTracker(len(val_loader), "  Validation")
        
        with torch.no_grad():
            for data, target in val_loader:
                with torch.cuda.amp.autocast():
                    output_tuple = model(data)
                    output = output_tuple[1]
                
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
                
                val_progress.update()
        
        val_accuracy = 100. * val_correct / val_total
        val_accuracy_history.append(val_accuracy)
        
        # Update learning rate
        scheduler.step(val_accuracy)
        
        epoch_time = time.time() - epoch_start_time
        
        # ========== EPOCH SUMMARY ==========
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Time: {epoch_time:.1f}s")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Train Accuracy: {train_accuracy:.2f}%")
        print(f"   Val Accuracy: {val_accuracy:.2f}%")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'train_loss': avg_train_loss,
                'config': model_config
            }, 'best_model_precise.pth')
            
            print(f"   💾 Saved BEST model (Accuracy: {val_accuracy:.2f}%)")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement for {patience_counter} epochs")
        
        # Early stopping check
        if patience_counter >= PATIENCE:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            break
    
    total_training_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_training_time/60:.1f} minutes")
    
    # ==================== FINAL EVALUATION ====================
    print("\n" + "=" * 70)
    print("🧪 FINAL MODEL EVALUATION")
    print("=" * 70)
    
    # Load best model
    if os.path.exists('best_model_precise.pth'):
        checkpoint = torch.load('best_model_precise.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"📥 Loaded best model from epoch {checkpoint['epoch'] + 1}")
        print(f"   Validation accuracy: {checkpoint['val_accuracy']:.2f}%")
    
    # Evaluate on test set
    model.eval()
    test_predictions = []
    test_targets = []
    
    test_progress = ProgressTracker(len(test_loader), "  Testing")
    
    with torch.no_grad():
        for data, target in test_loader:
            with torch.cuda.amp.autocast():
                output_tuple = model(data)
                output = output_tuple[1]
            
            _, predicted = output.max(1)
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
            
            test_progress.update()
    
    # Calculate metrics
    test_accuracy = accuracy_score(test_targets, test_predictions) * 100
    
    print(f"\n🎯 FINAL TEST RESULTS:")
    print(f"   Test Accuracy: {test_accuracy:.2f}%")
    print(f"   Best Val Accuracy: {best_val_accuracy:.2f}%")
    
    # Detailed classification report
    print("\n📈 CLASSIFICATION REPORT:")
    print(classification_report(test_targets, test_predictions, target_names=['Real', 'AI']))
    
    # Confusion matrix
    cm = confusion_matrix(test_targets, test_predictions)
    print("📊 CONFUSION MATRIX:")
    print(f"                   Predicted")
    print(f"                  Real    AI")
    print(f"Actual Real:    {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"        AI:     {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Save final model
    final_checkpoint = {
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'test_accuracy': test_accuracy,
        'best_val_accuracy': best_val_accuracy,
        'config': model_config,
        'training_time': total_training_time
    }
    
    torch.save(final_checkpoint, 'final_model_precise.pth')
    
    print(f"\n💾 Models saved:")
    print(f"   - best_model_precise.pth (Validation: {best_val_accuracy:.2f}%)")
    print(f"   - final_model_precise.pth (Test: {test_accuracy:.2f}%)")
    
    # Performance summary
    print("\n" + "=" * 70)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Total training time: {total_training_time/60:.1f} minutes")
    print(f"Epochs completed: {epoch+1}")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"GPU utilized: {torch.cuda.get_device_name(0)}")
    
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory used: {memory_used:.2f} GB")
    
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE - Ready for deployment!")
    print("=" * 70)

# ==================== RUN TRAINING ====================
if __name__ == "__main__":
    try:
        train_precise_fast()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
