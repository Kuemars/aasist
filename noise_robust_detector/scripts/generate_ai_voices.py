import requests
import soundfile as sf
import numpy as np
import librosa
import io
import time
import random

# Hugging Face TTS models - all free, no API key needed
TTS_MODELS = [
    "microsoft/speecht5_tts",
    "facebook/mms-tts-eng",
    "facebook/mms-tts-eng1",
    "facebook/mms-tts-eng2"
]

def generate_hf_tts(text, model_name):
    """Generate TTS using Hugging Face Inference API"""
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    
    payload = {
        "inputs": text,
        "parameters": {
            "length_penalty": 2.0,
            "repetition_penalty": 2.0,
            "do_sample": True,
            "temperature": 0.8
        }
    }
    
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        return response.content
    elif response.status_code == 503:
        # Model is loading, wait and retry
        print(f"⏳ Model loading, waiting 20 seconds...")
        time.sleep(20)
        return generate_hf_tts(text, model_name)
    else:
        print(f"❌ API Error {response.status_code}: {response.text[:100]}")
        return None

def normalize_audio(audio_bytes, target_duration=4.0, sr=16000):
    """Convert audio to 4 seconds with padding"""
    try:
        # Load audio from bytes
        audio, original_sr = sf.read(io.BytesIO(audio_bytes))
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
            
        # Resample to 16kHz if needed
        if original_sr != sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=sr)
        
        # Normalize to target duration
        target_samples = int(target_duration * sr)
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        else:
            padding = target_samples - len(audio)
            audio = np.pad(audio, (0, padding))
            
        return audio
    except Exception as e:
        print(f"Audio processing error: {e}")
        return None

# Diverse texts for TTS
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence advances rapidly each year.",
    "Voice recognition systems continue to improve constantly.",
    "Machine learning requires diverse training datasets.",
    "Digital signal processing enables clear audio quality.",
    "Natural language processing enhances communication.",
    "Cloud computing provides scalable infrastructure.",
    "Data security remains critically important always.",
    "User experience design creates intuitive interfaces.",
    "Real-time systems enable immediate responses quickly.",
]

current_count = 1400
voices_generated = 0
target_voices = 50  # How many new voices you want

print(f"🎯 Generating {target_voices} diverse Hugging Face TTS voices...")
print(f"🤖 Using {len(TTS_MODELS)} different TTS models")

for i in range(target_voices):
    model = random.choice(TTS_MODELS)
    text = random.choice(texts)
    
    print(f"🔊 Generating voice {i+1}/{target_voices} with {model.split('/')[-1]}...")
    
    try:
        audio_bytes = generate_hf_tts(text, model)
        
        if audio_bytes:
            normalized_audio = normalize_audio(audio_bytes)
            
            if normalized_audio is not None:
                output_file = f"ai_{current_count + i + 1:04d}.wav"
                sf.write(output_file, normalized_audio, 16000)
                
                voices_generated += 1
                print(f"✅ Voice {voices_generated} completed")
            else:
                print(f"❌ Failed to process audio")
        else:
            print(f"❌ API failed, trying different model")
            # Try a different model if one fails
            TTS_MODELS.remove(model)
            if not TTS_MODELS:
                print("❌ No working models left")
                break
        
        # Rate limiting
        time.sleep(2)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        continue

print(f"\n🎉 Generation complete!")
print(f"📊 Generated {voices_generated} new AI voices")
print(f"📈 Total AI voices: {current_count + voices_generated}")