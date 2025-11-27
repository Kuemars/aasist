import os
import sys
import torch
import soundfile as sf
import numpy as np
import random
from tqdm import tqdm
import time

# Configuration
TARGET_SR = 16000
TARGET_DURATION = 4.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)
OUTPUT_DIR = "data/raw/clean_ai"

def get_current_ai_count():
    """Get current number of AI voice files"""
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('ai_') and f.endswith('.wav')]
    return len(existing)

def generate_diverse_texts(num_texts=600):
    """Generate diverse text samples for TTS generation"""
    print("📝 Generating diverse text samples...")
    
    # More natural, conversational texts
    texts = [
        # Simple conversational
        "Hello, how are you doing today?",
        "The weather is quite nice outside.",
        "I enjoy reading books in my free time.",
        "Technology continues to advance rapidly.",
        "Learning new skills can be rewarding.",
        "The sound of rain is very peaceful.",
        "I appreciate simple everyday moments.",
        "Music brings people together nicely.",
        "Nature always finds ways to inspire.",
        "Coffee in the morning is wonderful.",
        
        # Slightly longer phrases
        "The system processes information efficiently and accurately.",
        "Modern algorithms analyze complex data patterns effectively.",
        "Digital interfaces have become more intuitive over time.",
        "Machine learning requires careful training and validation.",
        "Voice recognition technology has improved significantly.",
        "Artificial intelligence transforms many industries today.",
        "Data security remains critically important for systems.",
        "User experience design creates seamless interactions.",
        "Cloud computing enables scalable flexible solutions.",
        "Real-time processing allows immediate response feedback.",
        
        # Natural sounding statements
        "Every great discovery begins with simple curiosity.",
        "Through challenges we often find our true strength.",
        "City lights twinkle like stars at nighttime.",
        "Childhood memories filled with laughter and joy.",
        "Quiet moments often bring the deepest insights.",
        "Ocean waves bring a wonderful sense of calm.",
        "Adventure awaits around every new corner.",
        "Friendship is one of life's precious gifts.",
        "Morning sunlight filters through the window.",
        "Evening walks can be very refreshing.",
        
        # Educational content
        "Learning is a lifelong journey of discovery.",
        "Critical thinking helps analyze information.",
        "Communication skills are essential worldwide.",
        "Scientific research expands understanding.",
        "Historical knowledge provides context.",
        "Mathematical principles form foundations.",
        "Creative expression shares perspectives.",
        "Cultural diversity enriches experience.",
        "Environmental awareness is crucial.",
        "Health contributes to quality living.",
        
        # Professional scenarios
        "The report shows growth in sectors.",
        "Team collaboration leads to solutions.",
        "Customer feedback provides insights.",
        "Strategic planning ensures success.",
        "Effective leadership inspires teams.",
        "Project management requires coordination.",
        "Market analysis reveals opportunities.",
        "Quality assurance maintains standards.",
        "Development enhances career prospects.",
        "Networking creates valuable connections."
    ]
    
    # Expand with variations
    expanded_texts = []
    base_count = len(texts)
    
    for i in range(num_texts):
        base_text = texts[i % base_count]
        
        # Add variations
        if i < num_texts * 0.3:
            expanded_texts.append(base_text)
        elif i < num_texts * 0.6:
            expanded_texts.append(base_text.lower())
        else:
            expanded_texts.append(base_text.capitalize())
    
    random.shuffle(expanded_texts)
    print(f"✅ Generated {len(expanded_texts)} text samples")
    return expanded_texts

def install_kokoro():
    """Install Kokoro correctly"""
    print("🔧 Installing Kokoro-82M...")
    
    # Try different installation methods
    commands = [
        "pip install kokoro",
        "pip install git+https://github.com/hexgrad/Kokoro-82M",
        "pip install torchaudio --upgrade",
    ]
    
    for cmd in commands:
        print(f"   Trying: {cmd}")
        result = os.system(cmd)
        if result == 0:
            print(f"   ✅ Success with: {cmd}")
            return True
    
    print("❌ All installation methods failed")
    return False

def generate_with_kokoro(texts):
    """Generate AI voices using Kokoro-82M model"""
    print("🎤 Loading Kokoro-82M model...")
    
    # Try to import Kokoro
    try:
        # First try the likely import
        from kokoro import Kokoro
        print("✅ Kokoro imported successfully")
    except ImportError:
        print("❌ Kokoro not available, trying to install...")
        if not install_kokoro():
            return 0
        try:
            from kokoro import Kokoro
            print("✅ Kokoro imported after installation")
        except ImportError as e:
            print(f"❌ Still cannot import Kokoro: {e}")
            return 0
    
    try:
        # Initialize model (adjust based on actual API)
        # Kokoro might need different initialization
        model = Kokoro()
        print("✅ Kokoro-82M model initialized")
    except Exception as e:
        print(f"❌ Error initializing Kokoro: {e}")
        return 0
    
    current_count = get_current_ai_count()
    start_count = current_count + 1
    generated_count = 0
    
    print(f"🎵 Generating {len(texts)} AI voices with Kokoro...")
    
    for i, text in enumerate(tqdm(texts, desc="Kokoro Generation")):
        try:
            # Generate speech - adjust based on actual Kokoro API
            # This might need to be changed based on the actual library
            result = model.generate(text)
            
            # Handle different return types
            if isinstance(result, tuple):
                audio_array, sample_rate = result
            elif hasattr(result, 'audio') and hasattr(result, 'sample_rate'):
                audio_array = result.audio
                sample_rate = result.sample_rate
            else:
                audio_array = result
                sample_rate = 22050  # Default assumption
            
            # Convert to numpy if needed
            if torch.is_tensor(audio_array):
                audio_array = audio_array.cpu().numpy()
            
            # Ensure mono and right shape
            if len(audio_array.shape) > 1:
                audio_array = audio_array.flatten()
            
            # Resample to target rate if needed
            if sample_rate != TARGET_SR:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=TARGET_SR)
            
            # Normalize to target duration
            if len(audio_array) > TARGET_SAMPLES:
                # Trim from center
                start = (len(audio_array) - TARGET_SAMPLES) // 2
                audio_array = audio_array[start:start + TARGET_SAMPLES]
            else:
                # Pad with silence
                padding = TARGET_SAMPLES - len(audio_array)
                audio_array = np.pad(audio_array, (0, padding))
            
            # Save the file
            filename = f"ai_{start_count + i:04d}.wav"
            output_path = os.path.join(OUTPUT_DIR, filename)
            sf.write(output_path, audio_array, TARGET_SR)
            
            generated_count += 1
            
            # Progress update
            if generated_count % 50 == 0:
                print(f"   ✅ Generated {generated_count}/{len(texts)}")
                
        except Exception as e:
            print(f"❌ Error generating voice {i}: {e}")
            continue
    
    return generated_count

def robust_gtts_fallback(texts):
    """More robust gTTS fallback with better error handling"""
    print("🔄 Using robust gTTS fallback...")
    
    try:
        from gtts import gTTS
        import librosa
    except ImportError:
        print("❌ gTTS not available")
        return 0
    
    current_count = get_current_ai_count()
    start_count = current_count + 1
    generated_count = 0
    
    for i, text in enumerate(tqdm(texts, desc="gTTS Generation")):
        try:
            # Skip if text is too long for gTTS
            if len(text) > 200:
                continue
                
            # Generate speech with gTTS
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save temporary file
            temp_file = os.path.join(OUTPUT_DIR, f'temp_{i}.mp3')
            tts.save(temp_file)
            
            # Load and convert
            audio, sr = librosa.load(temp_file, sr=TARGET_SR)
            
            # Normalize length
            if len(audio) > TARGET_SAMPLES:
                audio = audio[:TARGET_SAMPLES]
            else:
                padding = TARGET_SAMPLES - len(audio)
                audio = np.pad(audio, (0, padding))
            
            # Save final file
            filename = f"ai_{start_count + generated_count:04d}.wav"
            output_path = os.path.join(OUTPUT_DIR, filename)
            sf.write(output_path, audio, TARGET_SR)
            
            generated_count += 1
            
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            # Longer delay to avoid rate limits
            time.sleep(1.0)
            
            # Progress
            if generated_count % 30 == 0:
                print(f"   ✅ Generated {generated_count}/{len(texts)}")
                time.sleep(5)  # Extra break every 30 files
                
        except Exception as e:
            print(f"❌ Error in gTTS {i}: {e}")
            time.sleep(10)  # Longer delay on error
            continue
    
    return generated_count

def main():
    print("🚀 GENERATING 600 AI VOICES WITH KOKORO-82M")
    print("=" * 50)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    current_count = get_current_ai_count()
    print(f"📊 Current AI voices: {current_count}")
    
    # Generate texts
    texts = generate_diverse_texts(600)
    
    # Try Kokoro first
    print("\n🎯 Attempting Kokoro-82M generation...")
    generated_count = generate_with_kokoro(texts)
    
    # Fallback if needed
    if generated_count < 300:  # If we got less than half
        print(f"\n⚠️  Kokoro only generated {generated_count}, using robust fallback...")
        additional_count = robust_gtts_fallback(texts[generated_count:])
        generated_count += additional_count
    
    # Final results
    final_count = get_current_ai_count()
    print(f"\n🎉 GENERATION COMPLETE!")
    print(f"📊 Generated {generated_count} new AI voices")
    print(f"📈 Total AI voices: {final_count}")
    print(f"⚖️  Dataset balance: 2000 real vs {final_count} AI")

if __name__ == "__main__":
    main()