import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import yaml
import time
import warnings
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================
# FIXED IMPORT PATHS
# ============================================
current_dir = os.path.dirname(os.path.abspath(__file__))
noise_robust_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(noise_robust_dir)
sys.path.append(project_root)

try:
    from models.AASIST import Model
    print(f"✅ AASIST import successful from {project_root}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# ============================================
# OPTIMIZED CONFIGURATION FOR RTX 4080 SUPER
# ============================================
SAMPLE_RATE = 16000
AUDIO_LENGTH = 64600  # 4.04 seconds at 16kHz
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.0001
USE_AMP = True
PATIENCE = 10
MIN_DELTA = 0.0001
USE_AUGMENTATION = True

# ============================================
# PATHS (RELATIVE TO noise_robust_detector/)
# ============================================
REAL_PATH = "data/raw/clean_real"
AI_PATH = "data/raw/clean_ai"
NOISE_PATH = "data/raw/noise_samples"
PRETRAINED_WEIGHTS = os.path.join(project_root, "models/weights/AASIST.pth")
CONFIG_PATH = os.path.join(project_root, "config/AASIST.conf")

# ============================================
# CONFIG LOADER WITH ERROR HANDLING
# ============================================
def get_aasist_config():
    """Load AASIST config with robust error handling"""
    default_config = {
        'nb_samp': 64600,
        'nb_classes': 2,
        'dim_emb': 192,
        'sr': 16000
    }
    
    if not os.path.exists(CONFIG_PATH):
        print(f"⚠️ Config file not found at {CONFIG_PATH}, using defaults")
        return default_config
    
    try:
        with open(CONFIG_PATH, 'r') as f:
            content = f.read()
            # Replace tabs with spaces to avoid YAML parsing errors
            content = content.replace('\t', '    ')
            config = yaml.safe_load(content)
            
            if config and 'model_config' in config:
                model_config = config['model_config']
            else:
                model_config = {}
                
    except Exception as e:
        print(f"⚠️ Error loading config: {e}, using defaults")
        model_config = {}
    
    # Ensure critical parameters are set
    model_config['nb_samp'] = model_config.get('nb_samp', 64600)
    model_config['nb_classes'] = model_config.get('nb_classes', 2)
    model_config['dim_emb'] = model_config.get('dim_emb', 192)
    model_config['sr'] = model_config.get('sr', 16000)
    
    print(f"📋 Model config: nb_classes={model_config['nb_classes']}, nb_samp={model_config['nb_samp']}")
    return model_config

# ============================================
# ENHANCED DATASET
# ============================================
class EnhancedAudioDataset(Dataset):
    def __init__(self, real_files, ai_files, train_mode=True, augment=False):
        self.files = []
        self.labels = []
        self.train_mode = train_mode
        self.augment = augment and train_mode
        
        for file in real_files:
            self.files.append(file)
            self.labels.append(0)
        
        for file in ai_files:
            self.files.append(file)
            self.labels.append(1)
        
        self.noise_samples = []
        if self.augment and os.path.exists(NOISE_PATH):
            noise_files = [os.path.join(NOISE_PATH, f) for f in os.listdir(NOISE_PATH) 
                          if f.endswith('.wav')][:20]
            for noise_file in noise_files:
                try:
                    noise, _ = librosa.load(noise_file, sr=SAMPLE_RATE)
                    if len(noise) >= AUDIO_LENGTH:
                        self.noise_samples.append(noise[:AUDIO_LENGTH])
                except:
                    continue
    
    def __len__(self):
        return len(self.files)
    
    def _augment_audio(self, audio):
        if not self.augment or len(self.noise_samples) == 0:
            return audio
        
        if np.random.random() < 0.2:
            noise = self.noise_samples[np.random.randint(0, len(self.noise_samples))]
            audio = audio + 0.005 * noise
        
        return audio
    
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        
        try:
            audio, sr = sf.read(audio_path)
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        except:
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        if len(audio) > AUDIO_LENGTH:
            audio = audio[:AUDIO_LENGTH]
        elif len(audio) < AUDIO_LENGTH:
            padding = AUDIO_LENGTH - len(audio)
            audio = np.pad(audio, (0, padding))
        
        if self.train_mode and self.augment:
            audio = self._augment_audio(audio)
        
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        audio_tensor = torch.FloatTensor(audio)
        label_tensor = torch.LongTensor([self.labels[idx]])
        
        return audio_tensor, label_tensor

# ============================================
# COLLATE FUNCTION
# ============================================
def collate_fn(batch):
    batch_size = len(batch)
    audio_batch = torch.zeros(batch_size, AUDIO_LENGTH, dtype=torch.float32)
    labels = torch.zeros(batch_size, dtype=torch.long)
    
    for i, (audio, label) in enumerate(batch):
        audio_batch[i] = audio
        labels[i] = label.squeeze()
    
    return audio_batch, labels

# ============================================
# EARLY STOPPING
# ============================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

# ============================================
# MAIN TRAINING FUNCTION
# ============================================
def enhanced_fine_tune_aasist():
    print("=" * 60)
    print("🚀 ENHANCED AASIST FINE-TUNING")
    print("=" * 60)
    
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    if device.type == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    
    # Load data
    real_files = [os.path.join(REAL_PATH, f) for f in os.listdir(REAL_PATH) 
                  if f.endswith('.wav')]
    ai_files = [os.path.join(AI_PATH, f) for f in os.listdir(AI_PATH) 
                if f.endswith('.wav')]
    
    print(f"\n📊 Dataset: {len(real_files)} real, {len(ai_files)} AI")
    
    # Split data
    real_train, real_temp = train_test_split(real_files, test_size=0.3, random_state=42)
    ai_train, ai_temp = train_test_split(ai_files, test_size=0.3, random_state=42)
    real_val, real_test = train_test_split(real_temp, test_size=0.5, random_state=42)
    ai_val, ai_test = train_test_split(ai_temp, test_size=0.5, random_state=42)
    
    print(f"📈 Split: Train={len(real_train)+len(ai_train)}, Val={len(real_val)+len(ai_val)}, Test={len(real_test)+len(ai_test)}")
    
    # Create datasets
    train_dataset = EnhancedAudioDataset(real_train, ai_train, train_mode=True, augment=USE_AUGMENTATION)
    val_dataset = EnhancedAudioDataset(real_val, ai_val, train_mode=False, augment=False)
    test_dataset = EnhancedAudioDataset(real_test, ai_test, train_mode=False, augment=False)
    
    # Create data loaders
    num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             collate_fn=collate_fn, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, 
                           collate_fn=collate_fn, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, 
                            collate_fn=collate_fn)
    
    # Initialize model
    print(f"\n🧠 Initializing model...")
    model_config = get_aasist_config()
    model = Model(model_config).to(device)
    
    # Load pre-trained weights
    if os.path.exists(PRETRAINED_WEIGHTS):
        print(f"📥 Loading weights from {PRETRAINED_WEIGHTS}")
        try:
            state_dict = torch.load(PRETRAINED_WEIGHTS, map_location=device)
            model.load_state_dict(state_dict)
            print("✅ Weights loaded")
        except Exception as e:
            print(f"⚠️ Error loading weights: {e}")
    else:
        print("⚠️ No pre-trained weights found")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scaler = GradScaler() if USE_AMP and device.type == 'cuda' else None
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
    
    print(f"\n⚙️ Config: Batch={BATCH_SIZE}, Epochs={EPOCHS}, LR={LEARNING_RATE}")
    
    # Training loop
    train_losses, val_losses, val_accuracies = [], [], []
    best_val_accuracy = 0
    best_model_state = None
    
    print(f"\n🎯 Starting training...")
    print("=" * 60)
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Training
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]', leave=False)
        
        for data, target in train_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            if USE_AMP and scaler is not None:
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
            
            running_loss += loss.item()
            train_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)[1]
                loss = criterion(output, target)
                val_running_loss += loss.item()
                
                preds = output.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = accuracy_score(val_targets, val_preds)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        scheduler.step(avg_val_loss)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
        
        epoch_time = time.time() - epoch_start
        print(f"📊 Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.4f}, Time={epoch_time:.1f}s")
        
        if early_stopping(avg_val_loss):
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Total time: {total_time/60:.1f} minutes")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_save_path = 'models/aasist_enhanced_fine_tuned.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Model saved to {model_save_path}")
    
    # Final test
    print(f"\n🧪 Testing...")
    model.eval()
    test_preds, test_targets = [], []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)[1]
            preds = output.argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
    
    test_accuracy = accuracy_score(test_targets, test_preds)
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Best Val Accuracy: {best_val_accuracy:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(test_targets, test_preds, target_names=['Real', 'AI']))
    
    print("=" * 60)
    return model

if __name__ == "__main__":
    enhanced_fine_tune_aasist()
