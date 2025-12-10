# save as create_balanced_augmented.py
import os
import numpy as np
import librosa
import soundfile as sf
import random
from tqdm import tqdm
from datasets import load_dataset

# Paths
REAL_SOURCE_DIR = "data/raw/clean_real"
AI_SOURCE_DIR = "data/raw/clean_ai"
AUGMENTED_OUTPUT_DIR = "data/augmented_balanced"
AUG_HUMAN_DIR = os.path.join(AUGMENTED_OUTPUT_DIR, "human")
AUG_AI_DIR = os.path.join(AUGMENTED_OUTPUT_DIR, "ai")

# Ensure output directories exist
os.makedirs(AUG_HUMAN_DIR, exist_ok=True)
os.makedirs(AUG_AI_DIR, exist_ok=True)

def load_noise_samples():
    """Load background noise for both human and AI"""
    print("📥 Loading noise samples...")
    noise_samples = []
    
    # Try HuggingFace first
    try:
        ds = load_dataset("Myrtle/CAIMAN-ASR-BackgroundNoise", split='train', streaming=True)
        for i, item in enumerate(ds):
            if i >= 50:  # Limit to 50 diverse noises
                break
            try:
                audio = item['audio']['array']
                sr = item['audio']['sampling_rate']
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                if len(audio) > 16000:
                    noise_samples.append(audio)
            except:
                continue
    except:
        # Fallback to local
        noise_dir = "data/raw/noise_samples"
        if os.path.exists(noise_dir):
            for f in os.listdir(noise_dir)[:50]:
                if f.endswith('.wav'):
                    try:
                        noise, sr = librosa.load(os.path.join(noise_dir, f), sr=16000)
                        if len(noise) > 16000:
                            noise_samples.append(noise)
                    except:
                        continue
    
    print(f"✅ Loaded {len(noise_samples)} noise samples")
    return noise_samples

def augment_human_voice(audio, sr, noise_samples):
    """Apply HUMAN-SPECIFIC mild augmentations"""
    augmented = audio.copy()
    
    # 1. Very mild time stretch (natural speaking speed variation)
    if random.random() < 0.3:
        stretch_rate = random.uniform(0.97, 1.03)  # Only ±3%
        stretched = librosa.effects.time_stretch(audio, rate=stretch_rate)
        # Maintain length
        if len(stretched) > len(audio):
            start = (len(stretched) - len(audio)) // 2
            stretched = stretched[start:start + len(audio)]
        else:
            pad_before = (len(audio) - len(stretched)) // 2
            pad_after = len(audio) - len(stretched) - pad_before
            stretched = np.pad(stretched, (pad_before, pad_after))
        augmented = stretched
    
    # 2. Tiny pitch shift (natural pitch drift)
    if random.random() < 0.2:
        pitch_steps = random.uniform(-0.5, 0.5)  # Only ±0.5 semitones
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_steps)
        if len(shifted) != len(audio):
            if len(shifted) > len(audio):
                shifted = shifted[:len(audio)]
            else:
                shifted = np.pad(shifted, (0, len(audio) - len(shifted)))
        augmented = shifted
    
    # 3. Background noise (same environments humans exist in)
    if noise_samples and random.random() < 0.4:
        noise = random.choice(noise_samples)
        if len(noise) < len(audio):
            repeats = int(np.ceil(len(audio) / len(noise)))
            noise = np.tile(noise, repeats)
        noise = noise[:len(audio)]
        noise_level = random.uniform(0.005, 0.02)  # Subtle background
        augmented = augmented + noise_level * noise
    
    # 4. Room reverb (natural acoustic)
    if random.random() < 0.2:
        reverb_length = int(sr * random.uniform(0.1, 0.3))  # 100-300ms
        impulse = np.exp(-np.linspace(0, 8, reverb_length))
        from scipy import signal
        reverberated = signal.convolve(augmented, impulse, mode='same')
        augmented = 0.8 * augmented + 0.2 * reverberated
    
    # 5. Breathing/mouth sounds (natural human artifacts)
    if random.random() < 0.1:
        # Add subtle breath-like noise
        breath = np.random.randn(len(audio)) * 0.002
        breath_freq = random.uniform(100, 300)
        t = np.arange(len(audio)) / sr
        breath *= (0.5 + 0.5 * np.sin(2 * np.pi * breath_freq * t))
        augmented += breath
    
    # Normalize
    augmented = augmented / (np.max(np.abs(augmented)) + 1e-8)
    return augmented

def augment_ai_voice(audio, sr, noise_samples):
    """Apply AI-SPECIFIC extreme augmentations"""
    augmented = audio.copy()
    
    # 1. Significant time stretch (unnatural playback speeds)
    if random.random() < 0.5:
        stretch_rate = random.choice([0.85, 0.9, 0.95, 1.05, 1.1, 1.15])
        stretched = librosa.effects.time_stretch(audio, rate=stretch_rate)
        if len(stretched) > len(audio):
            stretched = stretched[:len(audio)]
        else:
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)))
        augmented = stretched
    
    # 2. Significant pitch shift (unnatural voice changes)
    if random.random() < 0.5:
        pitch_steps = random.choice([-3, -2, 2, 3])
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_steps)
        if len(shifted) != len(audio):
            if len(shifted) > len(audio):
                shifted = shifted[:len(audio)]
            else:
                shifted = np.pad(shifted, (0, len(audio) - len(shifted)))
        augmented = shifted
    
    # 3. Background noise (can be louder/more prominent)
    if noise_samples and random.random() < 0.5:
        noise = random.choice(noise_samples)
        if len(noise) < len(audio):
            repeats = int(np.ceil(len(audio) / len(noise)))
            noise = np.tile(noise, repeats)
        noise = noise[:len(audio)]
        noise_level = random.uniform(0.01, 0.04)  # Can be louder
        augmented = augmented + noise_level * noise
    
    # 4. White noise (electronic/processing artifacts)
    if random.random() < 0.3:
        white_noise = np.random.randn(len(audio)) * random.uniform(0.001, 0.005)
        augmented += white_noise
    
    # 5. Telephone/bandwidth effect (processing artifact)
    if random.random() < 0.3:
        from scipy import signal
        nyquist = sr / 2
        # Telephone bandwidth
        lowcut = 300 / nyquist
        highcut = 3400 / nyquist
        b, a = signal.butter(4, [lowcut, highcut], btype='band')
        augmented = signal.filtfilt(b, a, augmented)
        # Add slight compression
        augmented = np.tanh(augmented * 0.8)
    
    # 6. EQ filtering (unnatural frequency response)
    if random.random() < 0.2:
        from scipy import signal
        nyquist = sr / 2
        # Random frequency cuts
        if random.random() < 0.5:
            lowcut = random.uniform(200, 500) / nyquist
            b, a = signal.butter(2, lowcut, btype='high')
            augmented = signal.filtfilt(b, a, augmented)
        if random.random() < 0.5:
            highcut = random.uniform(3000, 6000) / nyquist
            b, a = signal.butter(2, highcut, btype='low')
            augmented = signal.filtfilt(b, a, augmented)
    
    # Normalize
    augmented = augmented / (np.max(np.abs(augmented)) + 1e-8)
    return augmented

def create_balanced_dataset():
    """Create balanced augmented dataset (500 human + 500 AI)"""
    print("=" * 60)
    print("🎯 CREATING BALANCED AUGMENTED DATASET")
    print("=" * 60)
    
    # Load noise samples
    noise_samples = load_noise_samples()
    
    # Get file lists
    real_files = [os.path.join(REAL_SOURCE_DIR, f) for f in os.listdir(REAL_SOURCE_DIR) 
                  if f.endswith('.wav')]
    ai_files = [os.path.join(AI_SOURCE_DIR, f) for f in os.listdir(AI_SOURCE_DIR) 
                if f.endswith('.wav')]
    
    print(f"📊 Found {len(real_files)} human and {len(ai_files)} AI samples")
    print(f"🎯 Creating 500 augmented human and 500 augmented AI samples")
    
    # Select random subset of 500 each
    selected_real = random.sample(real_files, min(500, len(real_files)))
    selected_ai = random.sample(ai_files, min(500, len(ai_files)))
    
    # Augment human voices (mild)
    print("\n👤 Augmenting human voices (mild augmentations)...")
    human_count = 0
    for real_file in tqdm(selected_real, desc="Human voices"):
        try:
            audio, sr = librosa.load(real_file, sr=16000, duration=4.04)
            if len(audio) > 64600:
                audio = audio[:64600]
            elif len(audio) < 64600:
                audio = np.pad(audio, (0, 64600 - len(audio)))
            
            augmented = augment_human_voice(audio, sr, noise_samples)
            
            base_name = os.path.basename(real_file)
            aug_path = os.path.join(AUG_HUMAN_DIR, f"aug_{base_name}")
            sf.write(aug_path, augmented, sr)
            human_count += 1
            
        except Exception as e:
            print(f"⚠️ Error with {os.path.basename(real_file)}: {e}")
    
    # Augment AI voices (extreme)
    print("\n🤖 Augmenting AI voices (extreme augmentations)...")
    ai_count = 0
    for ai_file in tqdm(selected_ai, desc="AI voices"):
        try:
            audio, sr = librosa.load(ai_file, sr=16000, duration=4.04)
            if len(audio) > 64600:
                audio = audio[:64600]
            elif len(audio) < 64600:
                audio = np.pad(audio, (0, 64600 - len(audio)))
            
            augmented = augment_ai_voice(audio, sr, noise_samples)
            
            base_name = os.path.basename(ai_file)
            aug_path = os.path.join(AUG_AI_DIR, f"aug_{base_name}")
            sf.write(aug_path, augmented, sr)
            ai_count += 1
            
        except Exception as e:
            print(f"⚠️ Error with {os.path.basename(ai_file)}: {e}")
    
    # Summary
    print(f"\n✅ COMPLETED!")
    print(f"   Augmented human voices: {human_count}/500")
    print(f"   Augmented AI voices: {ai_count}/500")
    
    print(f"\n📈 FINAL BALANCED DATASET:")
    print(f"   Clean human voices: {len(real_files)}")
    print(f"   Clean AI voices: {len(ai_files)}")
    print(f"   Augmented human voices: {human_count} (mild)")
    print(f"   Augmented AI voices: {ai_count} (extreme)")
    total_samples = len(real_files) + len(ai_files) + human_count + ai_count
    print(f"   TOTAL SAMPLES: {total_samples}")
    clean_ratio = (len(real_files) + len(ai_files)) / total_samples * 100
    print(f"   Clean data: {clean_ratio:.1f}%")
    print(f"   Augmented data: {100 - clean_ratio:.1f}%")
    
    print(f"\n📁 Output directories:")
    print(f"   {AUG_HUMAN_DIR}")
    print(f"   {AUG_AI_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    create_balanced_dataset()