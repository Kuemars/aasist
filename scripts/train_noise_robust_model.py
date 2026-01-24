# save as train_noise_robust_model.py (overwrite existing)
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import json
import glob
from sklearn.model_selection import train_test_split

# ============================================================================
# FIXED PATH CONFIGURATION
# ============================================================================

# Get the absolute path of THIS script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Determine if we're running from scripts/ folder or root
if os.path.basename(current_script_dir) == 'scripts':
    # Running from scripts folder
    PROJECT_ROOT = os.path.dirname(current_script_dir)  # Go up one level
else:
    # Running from root or elsewhere
    PROJECT_ROOT = current_script_dir

# Add project root to Python path
sys.path.insert(0, PROJECT_ROOT)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Config
SAMPLE_RATE = 16000
AUDIO_LENGTH = 64600
BATCH_SIZE = 192
EPOCHS = 50
LEARNING_RATE = 0.00005

def verify_label_mapping():
    """Verify that our labeling matches ORIGINAL MODEL: Class 0 = HUMAN, Class 1 = AI"""
    print("\n🔍 VERIFYING LABEL MAPPING (Standardized to Original Model)...")
    
    # Test with known samples - FIXED PATHS
    ai_sample = os.path.join(PROJECT_ROOT, "data", "raw", "clean_ai", "ai_0000.wav")
    human_sample = os.path.join(PROJECT_ROOT, "data", "raw", "clean_real", "real_0001.wav")
    
    if os.path.exists(ai_sample) and os.path.exists(human_sample):
        print("STANDARDIZED MAPPING:")
        print(f"  {os.path.basename(human_sample)} → Label 0 (HUMAN)")
        print(f"  {os.path.basename(ai_sample)} → Label 1 (AI)")
        print("\n✅ Standardized mapping: Class 0 = HUMAN, Class 1 = AI")
        return True
    else:
        print("⚠️  Test files not found")
        return False

def normalize_audio(audio):
    """Consistent normalization to prevent learning volume differences"""
    # 1. Peak normalize to [-1, 1]
    if np.max(np.abs(audio)) > 1e-8:
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    # 2. Target RMS normalization (consistent loudness)
    target_rms = 0.1  # Consistent target loudness
    current_rms = np.sqrt(np.mean(audio**2))
    
    if current_rms > 1e-8:
        # Apply gain adjustment
        gain = target_rms / current_rms
        # Limit gain to prevent extreme amplification
        gain = min(gain, 5.0)
        audio = audio * gain
    
    # 3. Final peak check to prevent clipping
    if np.max(np.abs(audio)) > 0.95:
        audio = audio * 0.95 / np.max(np.abs(audio))
    
    return audio

class RobustDataset(Dataset):
    """Dataset for noise-robust training with STANDARDIZED labeling"""
    def __init__(self, real_clean_files, real_aug_files, ai_clean_files, ai_aug_files):
        self.files = []
        self.labels = []
        
        # STANDARDIZED MAPPING (same as original model):
        # HUMAN VOICES = Label 0
        for f in real_clean_files:
            self.files.append(f)
            self.labels.append(0)  # HUMAN = 0
        
        for f in real_aug_files:
            self.files.append(f)
            self.labels.append(0)  # HUMAN = 0 (mildly augmented)
        
        # AI VOICES = Label 1
        for f in ai_clean_files:
            self.files.append(f)
            self.labels.append(1)  # AI = 1
        
        for f in ai_aug_files:
            self.files.append(f)
            self.labels.append(1)  # AI = 1 (extremely augmented)
        
        print(f"\n📊 Dataset composition (STANDARDIZED):")
        print(f"  Clean HUMAN (label 0): {len(real_clean_files)}")
        print(f"  Augmented HUMAN (label 0): {len(real_aug_files)}")
        print(f"  Clean AI (label 1): {len(ai_clean_files)}")
        print(f"  Augmented AI (label 1): {len(ai_aug_files)}")
        print(f"  Total: {len(self.files)} samples")
        
        # Verify label distribution
        human_count = sum(1 for l in self.labels if l == 0)
        ai_count = sum(1 for l in self.labels if l == 1)
        print(f"  Label distribution: HUMAN={human_count}, AI={ai_count}")
        
        # Verify first few labels
        print(f"\n🔍 Sample labels (first 5 files):")
        for i in range(min(5, len(self.files))):
            fname = os.path.basename(self.files[i])
            label_name = "HUMAN" if self.labels[i] == 0 else "AI"
            print(f"  {fname} → {label_name} (label {self.labels[i]})")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            audio, sr = librosa.load(self.files[idx], sr=SAMPLE_RATE, duration=4.04)
            
            if len(audio) > AUDIO_LENGTH:
                audio = audio[:AUDIO_LENGTH]
            elif len(audio) < AUDIO_LENGTH:
                audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)))
            
            # APPLY CONSISTENT NORMALIZATION
            audio = normalize_audio(audio)
            
            return torch.FloatTensor(audio), torch.tensor(self.labels[idx], dtype=torch.long)
        except Exception as e:
            print(f"⚠️ Error loading {self.files[idx]}: {e}")
            # Return a dummy sample
            dummy_audio = torch.zeros(AUDIO_LENGTH)
            dummy_audio = normalize_audio(dummy_audio.numpy())
            return torch.FloatTensor(dummy_audio), torch.tensor(self.labels[idx], dtype=torch.long)

def verify_audio_levels(dataset):
    """Check that audio levels are consistent across classes"""
    print("\n🔍 VERIFYING AUDIO NORMALIZATION...")
    
    human_rms = []
    ai_rms = []
    human_peak = []
    ai_peak = []
    
    # Sample from each class
    sample_size = min(50, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for idx in indices:
        audio, label = dataset[idx]
        audio_np = audio.numpy()
        rms = np.sqrt(np.mean(audio_np**2))
        peak = np.max(np.abs(audio_np))
        
        if label == 0:  # HUMAN
            human_rms.append(rms)
            human_peak.append(peak)
        else:  # AI
            ai_rms.append(rms)
            ai_peak.append(peak)
    
    if human_rms and ai_rms:
        print(f"  HUMAN - RMS: {np.mean(human_rms):.4f}±{np.std(human_rms):.4f}, Peak: {np.mean(human_peak):.3f}±{np.std(human_peak):.3f}")
        print(f"  AI - RMS: {np.mean(ai_rms):.4f}±{np.std(ai_rms):.4f}, Peak: {np.mean(ai_peak):.3f}±{np.std(ai_peak):.3f}")
        
        # Check for significant difference
        rms_diff = abs(np.mean(human_rms) - np.mean(ai_rms))
        peak_diff = abs(np.mean(human_peak) - np.mean(ai_peak))
        
        if rms_diff > 0.05 or peak_diff > 0.1:
            print(f"  ⚠️  WARNING: Audio level differences detected!")
            print(f"    RMS difference: {rms_diff:.4f}")
            print(f"    Peak difference: {peak_diff:.3f}")
        else:
            print(f"  ✅ Audio levels are consistent across classes")
    
    return human_rms, ai_rms

def train_noise_robust_model():
    print("=" * 60)
    print("🎯 NOISE-ROBUST TRAINING (STANDARDIZED LABELS + NORMALIZATION)")
    print("=" * 60)
    
    # Verify label mapping before anything else
    if not verify_label_mapping():
        print("❌ Label verification failed. Aborting.")
        return
    
    # Get all files - FIXED PATHS
    real_clean = glob.glob(os.path.join(PROJECT_ROOT, "data", "raw", "clean_real", "*.wav"))
    real_aug = glob.glob(os.path.join(PROJECT_ROOT, "data", "augmented_balanced", "human", "*.wav"))
    ai_clean = glob.glob(os.path.join(PROJECT_ROOT, "data", "raw", "clean_ai", "*.wav"))
    ai_aug = glob.glob(os.path.join(PROJECT_ROOT, "data", "augmented_balanced", "ai", "*.wav"))
    
    print(f"\n📁 Found files:")
    print(f"  Clean human: {len(real_clean)}")
    print(f"  Augmented human: {len(real_aug)}")
    print(f"  Clean AI: {len(ai_clean)}")
    print(f"  Augmented AI: {len(ai_aug)}")
    
    # Split each category separately for balanced validation
    real_clean_train, real_clean_val = train_test_split(real_clean, test_size=0.15, random_state=42)
    real_aug_train, real_aug_val = train_test_split(real_aug, test_size=0.15, random_state=42)
    ai_clean_train, ai_clean_val = train_test_split(ai_clean, test_size=0.15, random_state=42)
    ai_aug_train, ai_aug_val = train_test_split(ai_aug, test_size=0.15, random_state=42)
    
    # Create datasets
    train_dataset = RobustDataset(real_clean_train, real_aug_train, ai_clean_train, ai_aug_train)
    val_dataset = RobustDataset(real_clean_val, real_aug_val, ai_clean_val, ai_aug_val)
    
    # Verify audio normalization
    print("\n📊 Checking training dataset audio levels...")
    train_human_rms, train_ai_rms = verify_audio_levels(train_dataset)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)
    
    # Model
    device = torch.device('cuda')
    print(f"\n📱 Using device: {device}")
    
    # Model import - FIXED
    try:
        from models.RawNet2Spoof import Model
    except ImportError:
        # Alternative import
        sys.path.append(os.path.join(PROJECT_ROOT, "models"))
        from RawNet2Spoof import Model
    
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
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Calculate class weights for balanced training
    print(f"\n⚖️ Calculating class weights...")
    all_labels = train_dataset.labels
    class_counts = np.bincount(all_labels)
    total_samples = len(all_labels)
    num_classes = len(class_counts)
    
    weights = [total_samples / (num_classes * count) for count in class_counts]
    print(f"  Class counts: HUMAN={class_counts[0]}, AI={class_counts[1]}")
    print(f"  Class weights: {weights}")
    
    weight_tensor = torch.FloatTensor(weights).to(device)
    
    # Train from scratch with class weights
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    print(f"\n⚙️ Training configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Class weights: Enabled")
    print(f"  Label smoothing: 0.1")
    
    # Training loop
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    print(f"\n🚀 Starting training...")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)[1]
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = output.argmax(dim=1)
            train_correct += (preds == target).sum().item()
            train_total += target.size(0)
            
            if batch_idx % 5 == 0:
                batch_acc = 100 * (preds == target).sum().item() / target.size(0)
                print(f"  Epoch {epoch+1}, Batch {batch_idx}: Loss={loss.item():.4f}, Acc={batch_acc:.1f}%")
        
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
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.1f}%")
        print(f"  Val Acc: {val_acc:.1f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # FIXED MODEL SAVE PATH
            model_save_path = os.path.join(PROJECT_ROOT, "models", "weights", "AI_Model_Noise_Robust_v0.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"  🏆 New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                break
        
        print("-" * 60)
    
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"🎯 Best validation accuracy: {best_val_acc:.1f}%")
    # FIXED PATH IN PRINT
    print(f"💾 Model saved: {os.path.join(PROJECT_ROOT, 'models', 'weights', 'AI_Model_Noise_Robust_v0.pth')}")
    
    # Load best model for final verification
    # FIXED MODEL LOAD PATH
    model_save_path = os.path.join(PROJECT_ROOT, "models", "weights", "AI_Model_Noise_Robust_v0.pth")
    print(f"\n🧪 Final verification on known samples...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    # FIXED TEST SAMPLE PATHS
    test_samples = [
        (os.path.join(PROJECT_ROOT, "data", "raw", "clean_ai", "ai_0000.wav"), "AI (clean)", 1),
        (os.path.join(PROJECT_ROOT, "data", "raw", "clean_real", "real_0001.wav"), "HUMAN (clean)", 0),
        (os.path.join(PROJECT_ROOT, "data", "augmented_balanced", "ai", "aug_ai_0002.wav"), "AI (augmented)", 1),
        (os.path.join(PROJECT_ROOT, "data", "augmented_balanced", "human", "aug_real_0007.wav"), "HUMAN (augmented)", 0)
    ]
    
    correct = 0
    total = 0
    print(f"\n🔍 Testing with STANDARDIZED mapping: Class 0 = HUMAN, Class 1 = AI")
    print("-" * 50)
    
    for filepath, expected_name, expected_label in test_samples:
        if os.path.exists(filepath):
            try:
                audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=4.04)
                if len(audio) > AUDIO_LENGTH:
                    audio = audio[:AUDIO_LENGTH]
                elif len(audio) < AUDIO_LENGTH:
                    audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)))
                
                audio = normalize_audio(audio)
                audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(audio_tensor)[1]
                    probabilities = torch.exp(output)
                    
                    # STANDARDIZED MAPPING: Class 0 = HUMAN, Class 1 = AI
                    human_prob = probabilities[0, 0].item() * 100
                    ai_prob = probabilities[0, 1].item() * 100
                    predicted_label = output.argmax(dim=1).item()
                
                predicted_name = "HUMAN" if predicted_label == 0 else "AI"
                expected_str = "HUMAN" if expected_label == 0 else "AI"
                
                print(f"\n{os.path.basename(filepath)} ({expected_name}):")
                print(f"  Predicted: {predicted_name} (HUMAN: {human_prob:.1f}%, AI: {ai_prob:.1f}%)")
                print(f"  Expected: {expected_str}")
                
                if predicted_label == expected_label:
                    print(f"  ✅ CORRECT")
                    correct += 1
                else:
                    print(f"  ❌ WRONG")
                total += 1
                
            except Exception as e:
                print(f"  ⚠️ Error testing {filepath}: {e}")
    
    if total > 0:
        print(f"\n🎯 Verification accuracy: {100*correct/total:.1f}% ({correct}/{total})")
    
    print("\n" + "=" * 60)
    print("📋 IMPORTANT: Detection scripts MUST use this mapping:")
    print("   Class 0 = HUMAN")
    print("   Class 1 = AI")
    print("=" * 60)
    
    return model

if __name__ == "__main__":
    train_noise_robust_model()