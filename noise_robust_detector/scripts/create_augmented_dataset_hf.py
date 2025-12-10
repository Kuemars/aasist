# save as create_augmented_dataset_hf.py
import os
import numpy as np
import librosa
import soundfile as sf
import random
from tqdm import tqdm
from datasets import load_dataset

# Paths
AI_SOURCE_DIR = "data/raw/clean_ai"
AUGMENTED_OUTPUT_DIR = "data/augmented"

# Ensure output directory exists
os.makedirs(AUGMENTED_OUTPUT_DIR, exist_ok=True)

def load_huggingface_noise_dataset():
    """Load background noise from HuggingFace dataset"""
    print("📥 Loading HuggingFace noise dataset...")
    
    try:
        # Load the dataset
        ds = load_dataset("Myrtle/CAIMAN-ASR-BackgroundNoise")
        
        # Get audio samples
        noise_samples = []
        total_duration = 0
        
        if 'train' in ds:
            split = ds['train']
        else:
            split = ds['audio'] if 'audio' in ds else list(ds.values())[0]
        
        # Process up to 100 noise samples (to avoid memory issues)
        for i, item in enumerate(split):
            if i >= 100:  # Limit to 100 diverse noise samples
                break
            
            try:
                # Get audio array
                if 'audio' in item:
                    audio = item['audio']['array']
                    sr = item['audio']['sampling_rate']
                elif 'array' in item:
                    audio = item['array']
                    sr = 16000  # Assume 16kHz
                else:
                    continue
                
                # Resample to 16kHz if needed
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                
                # Store
                noise_samples.append(audio)
                total_duration += len(audio) / 16000  # Duration in seconds
                
            except Exception as e:
                print(f"⚠️ Could not process noise sample {i}: {e}")
                continue
        
        print(f"✅ Loaded {len(noise_samples)} noise samples ({total_duration:.1f} seconds total)")
        return noise_samples
        
    except Exception as e:
        print(f"❌ Failed to load HuggingFace dataset: {e}")
        print("🔄 Falling back to local noise samples...")
        return load_local_noise_samples()

def load_local_noise_samples():
    """Fallback: Load noise samples from local directory"""
    noise_samples = []
    noise_dir = "data/raw/noise_samples"
    
    if os.path.exists(noise_dir):
        print(f"📁 Loading local noise samples from {noise_dir}")
        for f in os.listdir(noise_dir)[:50]:  # Use first 50
            if f.endswith(('.wav', '.mp3', '.flac')):
                try:
                    noise, sr = librosa.load(os.path.join(noise_dir, f), sr=16000)
                    if len(noise) > 16000:  # At least 1 second
                        noise_samples.append(noise)
                except:
                    continue
    
    print(f"✅ Loaded {len(noise_samples)} local noise samples")
    return noise_samples

def add_white_noise(audio, noise_level=0.005):
    """Add random white noise"""
    noise = np.random.randn(len(audio))
    return audio + noise_level * noise

def time_stretch_fixed(audio, rate=0.9):
    """Time stretch maintaining exact length"""
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    
    # Ensure same length
    if len(stretched) > len(audio):
        start = (len(stretched) - len(audio)) // 2
        stretched = stretched[start:start + len(audio)]
    else:
        pad_before = (len(audio) - len(stretched)) // 2
        pad_after = len(audio) - len(stretched) - pad_before
        stretched = np.pad(stretched, (pad_before, pad_after), mode='constant')
    
    return stretched

def pitch_shift_fixed(audio, sr=16000, n_steps=2):
    """Pitch shift maintaining length"""
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    if len(shifted) != len(audio):
        if len(shifted) > len(audio):
            shifted = shifted[:len(audio)]
        else:
            shifted = np.pad(shifted, (0, len(audio) - len(shifted)))
    
    return shifted

def add_background_noise(audio, noise_sample, noise_level=0.01):
    """Add background noise from sample"""
    # Trim or loop noise to match audio length
    if len(noise_sample) < len(audio):
        repeats = int(np.ceil(len(audio) / len(noise_sample)))
        noise_sample = np.tile(noise_sample, repeats)
    
    noise_sample = noise_sample[:len(audio)]
    return audio + noise_level * noise_sample

def apply_telephone_effect(audio, sr=16000):
    """Simulate telephone/mobile audio quality"""
    from scipy import signal
    
    # Telephone bandwidth (300Hz - 3400Hz)
    nyquist = sr / 2
    lowcut = 300 / nyquist
    highcut = 3400 / nyquist
    
    # Bandpass filter
    b, a = signal.butter(4, [lowcut, highcut], btype='band')
    audio = signal.filtfilt(b, a, audio)
    
    # Add slight compression
    audio = np.tanh(audio * 0.8)
    
    return audio

def apply_random_eq(audio, sr=16000):
    """Apply random equalization"""
    from scipy import signal
    
    # Random frequency cuts
    nyquist = sr / 2
    
    # Random low cut (simulate cheap mic)
    if random.random() < 0.3:
        lowcut = random.uniform(100, 400) / nyquist
        b, a = signal.butter(2, lowcut, btype='high')
        audio = signal.filtfilt(b, a, audio)
    
    # Random high cut (simulate bandwidth limit)
    if random.random() < 0.3:
        highcut = random.uniform(3000, 7000) / nyquist
        b, a = signal.butter(2, highcut, btype='low')
        audio = signal.filtfilt(b, a, audio)
    
    return audio

def add_room_reverb(audio, sr=16000):
    """Add simple room reverb effect"""
    # Simple impulse response simulation
    reverb_length = int(sr * 0.3)  # 300ms reverb
    impulse = np.exp(-np.linspace(0, 10, reverb_length))
    impulse = impulse / np.max(impulse)
    
    # Convolve with impulse response
    from scipy import signal
    reverberated = signal.convolve(audio, impulse, mode='same')
    
    # Mix dry and wet
    mix = 0.7 * audio + 0.3 * reverberated
    return mix / (np.max(np.abs(mix)) + 1e-8)

def create_augmented_version(audio, sr, noise_samples, aug_id):
    """Create one augmented version with realistic distortions"""
    augmented = audio.copy()
    
    # Apply realistic audio degradations (in sequence)
    transformations = []
    
    # 1. Time/Pitch modifications (simulate playback speed changes)
    if random.random() < 0.4:
        stretch_rate = random.choice([0.85, 0.9, 0.95, 1.05, 1.1])
        augmented = time_stretch_fixed(augmented, rate=stretch_rate)
        transformations.append(f"time_{stretch_rate}")
    
    if random.random() < 0.4:
        pitch_steps = random.choice([-3, -2, -1, 1, 2, 3])
        augmented = pitch_shift_fixed(augmented, sr=sr, n_steps=pitch_steps)
        transformations.append(f"pitch_{pitch_steps}")
    
    # 2. Background noise (realistic environments)
    if noise_samples and random.random() < 0.5:
        noise = random.choice(noise_samples)
        noise_level = random.uniform(0.005, 0.03)  # Varying noise levels
        augmented = add_background_noise(augmented, noise, noise_level=noise_level)
        transformations.append(f"noise_{noise_level:.3f}")
    
    # 3. White noise (electronic interference)
    if random.random() < 0.2:
        noise_level = random.uniform(0.001, 0.008)
        augmented = add_white_noise(augmented, noise_level=noise_level)
        transformations.append(f"white_{noise_level:.3f}")
    
    # 4. Audio quality degradation
    if random.random() < 0.3:
        augmented = apply_random_eq(augmented, sr)
        transformations.append("eq")
    
    if random.random() < 0.2:
        augmented = apply_telephone_effect(augmented, sr)
        transformations.append("telephone")
    
    if random.random() < 0.1:
        augmented = add_room_reverb(augmented, sr)
        transformations.append("reverb")
    
    # Normalize to prevent clipping
    augmented = augmented / (np.max(np.abs(augmented)) + 1e-8)
    
    return augmented, transformations

def create_augmented_dataset():
    """Create augmented versions of AI voices with realistic noise"""
    print("=" * 60)
    print("🔊 CREATING REALISTIC NOISE-ROBUST DATASET")
    print("=" * 60)
    
    # Load noise samples from HuggingFace
    noise_samples = load_huggingface_noise_dataset()
    
    # Get all AI files
    ai_files = []
    for f in os.listdir(AI_SOURCE_DIR):
        if f.endswith('.wav'):
            ai_files.append(os.path.join(AI_SOURCE_DIR, f))
    
    print(f"📊 Found {len(ai_files)} AI voice samples")
    print(f"🎯 Creating 3 realistic augmented versions per sample")
    
    # Track transformations for analysis
    transformation_stats = {}
    
    # Process each AI file
    processed_count = 0
    for ai_file in tqdm(ai_files, desc="Augmenting AI voices"):
        try:
            # Load original
            audio, sr = librosa.load(ai_file, sr=16000, duration=4.04)
            
            # Ensure correct length
            if len(audio) > 64600:
                audio = audio[:64600]
            elif len(audio) < 64600:
                audio = np.pad(audio, (0, 64600 - len(audio)))
            
            base_name = os.path.basename(ai_file)
            
            # Create 3 different augmented versions
            for aug_id in range(3):
                # Apply realistic augmentations
                augmented, transformations = create_augmented_version(audio, sr, noise_samples, aug_id)
                
                # Track transformation statistics
                for trans in transformations:
                    transformation_stats[trans] = transformation_stats.get(trans, 0) + 1
                
                # Save
                aug_filename = f"aug_{aug_id}_{base_name}"
                aug_path = os.path.join(AUGMENTED_OUTPUT_DIR, aug_filename)
                sf.write(aug_path, augmented, sr)
            
            processed_count += 1
            
        except Exception as e:
            print(f"\n⚠️ Error processing {ai_file}: {e}")
            continue
    
    # Print statistics
    print(f"\n✅ AUGMENTATION COMPLETE!")
    print(f"   Processed: {processed_count}/{len(ai_files)} files")
    print(f"   Created: {processed_count * 3} augmented files")
    print(f"   Location: {AUGMENTED_OUTPUT_DIR}")
    
    print(f"\n📊 TRANSFORMATION STATISTICS:")
    for trans, count in sorted(transformation_stats.items()):
        percentage = (count / (processed_count * 3)) * 100
        print(f"   {trans:20} {count:6d} times ({percentage:.1f}%)")
    
    # Dataset summary
    total_ai_files = len(ai_files) + (processed_count * 3)
    print(f"\n📈 FINAL DATASET COMPOSITION:")
    print(f"   Original AI voices: {len(ai_files)}")
    print(f"   Augmented AI voices: {processed_count * 3}")
    print(f"   Total AI voices: {total_ai_files}")
    print(f"   Real voices: 2000")
    print(f"   TOTAL SAMPLES: {total_ai_files + 2000}")
    print(f"\n⚠️  Note: Real voices remain clean. Only AI voices were augmented.")
    print("=" * 60)

if __name__ == "__main__":
    create_augmented_dataset()