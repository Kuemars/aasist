import os
import sys
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
    """Generate diverse text samples for TTS"""
    print("📝 Generating diverse text samples...")
    
    texts = [
        "Hello, how are you doing today?",
        "The weather is quite nice outside.",
        "I enjoy reading books in my free time.",
        "Technology continues to advance rapidly.",
        "Learning new skills can be rewarding.",
        "The system processes information efficiently.",
        "Modern algorithms analyze complex data patterns.",
        "Digital interfaces have become more intuitive.",
        "Machine learning requires careful training.",
        "Voice recognition has improved significantly.",
        "Artificial intelligence transforms industries.",
        "Data security remains critically important.",
        "User experience design creates seamless interactions.",
        "Cloud computing enables scalable solutions.",
        "Real-time processing allows immediate response.",
        "Natural language processing enables better communication.",
        "Computer vision systems can recognize objects accurately.",
        "Deep learning models require substantial computational resources.",
        "The internet connects people across the globe instantly.",
        "Mobile applications provide convenience and accessibility.",
        "Software development follows agile methodologies.",
        "Data analysis reveals valuable insights and patterns.",
        "Cybersecurity measures protect sensitive information.",
        "Automation increases efficiency and reduces errors.",
        "Virtual reality creates immersive digital experiences.",
        "Blockchain technology ensures secure transactions.",
        "Quantum computing promises revolutionary capabilities.",
        "Robotics automates complex physical tasks.",
        "Augmented reality overlays digital information.",
        "Internet of Things connects everyday devices.",
        "Five factor authentication provides enhanced security.",
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!",
        "Waltz, bad nymph, for quick jigs vex.",
        "Glib jocks quiz nymph to vex dwarf.",
        "Sphinx of black quartz, judge my vow.",
        "The five boxing wizards jump quickly.",
        "Bright vixens jump; dozy fowl quack.",
        "Jackdaws love my big sphinx of quartz.",
    ]
    
    # Expand the list to reach desired number
    expanded_texts = []
    while len(expanded_texts) < num_texts:
        for text in texts:
            if len(expanded_texts) >= num_texts:
                break
            expanded_texts.append(text)
    
    random.shuffle(expanded_texts)
    print(f"✅ Generated {len(expanded_texts)} text samples")
    return expanded_texts

def normalize_audio(audio_data, sr):
    """Normalize audio to target specifications"""
    # Ensure mono
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Resample if needed
    if sr != TARGET_SR:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=TARGET_SR)
    
    # Normalize to target duration
    if len(audio_data) > TARGET_SAMPLES:
        audio_data = audio_data[:TARGET_SAMPLES]
    else:
        padding = TARGET_SAMPLES - len(audio_data)
        audio_data = np.pad(audio_data, (0, padding))
    
    return audio_data

def get_prebuilt_voice_ids():
    """Return voice IDs for prebuilt voices that should work"""
    return {
        "Rachel": "21m00Tcm4TlvDq8ikWAM",  # Female voice 1
        "Domi": "AZnzlk1XvdvUeBnXmlld",    # Female voice 2  
        "Bella": "EXAVITQu4vr4xnSDxMaL",    # Female voice 3
        "Antoni": "ErXwobaYiN019PkySvjV",   # Male voice 1
        "Elli": "MF3mGyEYCl7XYWbV9V6O",     # Female voice 4
        "Josh": "TxGEqnHWrfWFTfGW9XjX",     # Male voice 2
        "Arnold": "VR6AewLTigWG4xSOukaG",   # Male voice 3
        "Adam": "pNInz6obpgDQGcFmaJgB",     # Male voice 4
        "Sam": "yoZ06aMxZJJ28mfd3POQ",      # Male voice 5
        "Dorothy": "ThT5KcBeYPX3keUQqHPh",  # Female voice 5
        "Charlotte": "XB0fDUnXU5powFXDhCwa", # Female voice 6
        "Alice": "Xb7hH8MSUJpSbSDYk0k2",    # Female voice 7
        "Matilda": "XrExE9yKIg1WjnnlVkGX",  # Female voice 8
        "Freya": "jsCqWAovK2LkecY7zXl4",    # Female voice 9
        "Grace": "oWAxZDx7w5VEj9dNgTKM",    # Female voice 10
        "Daniel": "onwK4e9ZLuTAKqWW03F9",   # Male voice 6
        "Liam": "TX3LPaxmHKxFdv7VOQHJ",     # Male voice 7
        "Callum": "N2lVS1w4EtoT3dr4eOWO",   # Male voice 8
        "Fin": "D38z5RcWu1voky8WS1ja",      # Male voice 9
        "George": "JBFqnCBsd6RMkjVDRZzb",   # Male voice 10
    }

def generate_with_elevenlabs_fixed(texts, api_key):
    """Generate AI voices using specific voice IDs"""
    print("🎤 Initializing ElevenLabs...")
    
    try:
        from elevenlabs import generate, save, set_api_key
        print("✅ ElevenLabs imported successfully")
    except ImportError:
        print("❌ ElevenLabs not available, installing...")
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "install", "elevenlabs"], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("Failed to install ElevenLabs")
        from elevenlabs import generate, save, set_api_key
        print("✅ ElevenLabs imported after installation")
    
    # Set API key
    set_api_key(api_key)
    
    current_count = get_current_ai_count()
    start_count = current_count + 1
    generated_count = 0
    
    print(f"🎵 Generating {len(texts)} AI voices with ElevenLabs...")
    
    # Get prebuilt voice IDs
    voice_ids = get_prebuilt_voice_ids()
    voice_names = list(voice_ids.keys())
    
    print(f"🎙️  Available voices: {', '.join(voice_names)}")
    
    # Test which voices actually work
    working_voices = test_voices(voice_ids, api_key)
    
    if not working_voices:
        print("❌ No working voices found! Please check your API key and quota.")
        return 0
    
    print(f"✅ Working voices: {', '.join(working_voices.keys())}")
    
    for i, text in enumerate(tqdm(texts, desc="ElevenLabs Generation")):
        try:
            # Rotate through working voices for diversity
            voice_name = list(working_voices.keys())[i % len(working_voices)]
            voice_id = working_voices[voice_name]
            
            # Use turbo model for free tier
            model = "eleven_turbo_v2"
            
            # Generate audio using specific voice ID
            audio = generate(
                text=text,
                voice=voice_id,  # Use voice ID instead of name
                model=model
            )
            
            # Save temporary file
            temp_file = os.path.join(OUTPUT_DIR, f"temp_eleven_{i}.wav")
            save(audio, temp_file)
            
            # Load and process the audio
            audio_data, sr = sf.read(temp_file)
            
            # Normalize audio to our specifications
            audio_data = normalize_audio(audio_data, sr)
            
            # Save final file with proper naming
            filename = f"ai_{start_count + i:04d}.wav"
            output_path = os.path.join(OUTPUT_DIR, filename)
            sf.write(output_path, audio_data, TARGET_SR)
            
            generated_count += 1
            
            # Cleanup temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            # Progress updates
            if generated_count % 10 == 0:
                print(f"   ✅ Generated {generated_count}/{len(texts)} (Voice: {voice_name})")
            
            # Rate limiting
            time.sleep(1.5)
                
        except Exception as e:
            print(f"❌ Error generating voice {i} with {voice_name}: {e}")
            
            # Remove failing voice from working voices
            if voice_name in working_voices:
                del working_voices[voice_name]
                print(f"   🗑️  Removed '{voice_name}' from working voices")
            
            # If no voices left, stop
            if not working_voices:
                print("❌ No more working voices available!")
                break
            
            print("   ⏳ Waiting 3 seconds before continuing...")
            time.sleep(3)
            continue
    
    return generated_count

def test_voices(voice_ids, api_key):
    """Test which voices actually work"""
    from elevenlabs import generate, save, set_api_key
    set_api_key(api_key)
    
    print("🧪 Testing which voices work...")
    working_voices = {}
    
    # Test a subset of voices first
    test_voices = list(voice_ids.items())[:8]  # Test first 8 to save time
    
    for voice_name, voice_id in test_voices:
        try:
            # Generate a very short test
            audio = generate(
                text="Test voice",
                voice=voice_id,
                model="eleven_turbo_v2"
            )
            
            # Try to save it
            test_file = f"test_{voice_name}.wav"
            save(audio, test_file)
            
            # Check if file was created
            if os.path.exists(test_file) and os.path.getsize(test_file) > 1000:
                working_voices[voice_name] = voice_id
                print(f"   ✅ {voice_name} works!")
                os.remove(test_file)
            else:
                if os.path.exists(test_file):
                    os.remove(test_file)
                print(f"   ❌ {voice_name} failed (empty file)")
                
        except Exception as e:
            print(f"   ❌ {voice_name} failed: {str(e)[:100]}...")
            continue
    
    return working_voices

def main():
    print("🚀 ELEVENLABS VOICE GENERATION (FIXED VOICE IDs)")
    print("=" * 60)
    print("💡 Using specific voice IDs to ensure diversity")
    print("🔗 Get API key: https://elevenlabs.io")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get current status
    current_count = get_current_ai_count()
    print(f"📊 Current AI voices: {current_count}")
    
    # Calculate needed voices
    target_count = 1400
    needed_count = target_count - current_count
    
    if needed_count <= 0:
        print("✅ Dataset already balanced!")
        return
    
    print(f"🎯 Need to generate: {needed_count} more voices")
    print(f"📈 Target total: {target_count} AI voices")
    
    # Get API key
    api_key = input("\n🔑 Enter your ElevenLabs API key: ").strip()
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return
    
    # Generate diverse texts
    texts = generate_diverse_texts(needed_count)
    
    print(f"\n🎵 Starting ElevenLabs generation...")
    print("💡 Using specific voice IDs to avoid 'Roger' issue")
    print("⏰ Estimated time: {:.1f} minutes".format(needed_count * 2.5 / 60))
    
    # Generate voices
    generated_count = generate_with_elevenlabs_fixed(texts, api_key)
    
    # Final status
    final_count = get_current_ai_count()
    
    print(f"\n" + "=" * 50)
    print("🎉 GENERATION COMPLETE!")
    print(f"📊 Generated: {generated_count} new voices")
    print(f"📈 Total AI voices: {final_count}")
    print(f"🎯 Target: {target_count}")
    print(f"📊 Real voices: 2000")
    print(f"⚖️ Balance ratio: 1:{final_count/2000:.2f}")
    
    if final_count >= target_count:
        print("✅ DATASET BALANCED - READY FOR TRAINING! 🚀")
    else:
        print(f"⚠️  Still need {target_count - final_count} more voices")

if __name__ == "__main__":
    main()