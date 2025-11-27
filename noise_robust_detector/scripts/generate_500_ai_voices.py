import os
import sys
import time
import random
from gtts import gTTS
import soundfile as sf
import librosa
import numpy as np

# Configuration
TARGET_SR = 16000
TARGET_DURATION = 4.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)
OUTPUT_DIR = "data/raw/clean_ai"

def get_current_ai_count():
    """Get current number of AI voice files"""
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('ai_') and f.endswith('.wav')]
    return len(existing)

def generate_diverse_texts(num_texts=500):
    """Generate diverse text samples for AI voice generation"""
    print("📝 Generating diverse text samples...")
    
    # Base templates for variety
    templates = [
        "The {} {} across the {} while {}",
        "In the {}, the {} began to {} with {}",
        "{} and {} worked together to {} the {}",
        "The {} of {} created a {} that {}",
        "When {} met {}, they discovered {} in the {}",
        "Through {} and {}, the {} achieved {}",
        "The {} transformed into {} during the {}",
        "{} explored the {} to find {} and {}",
        "During the {}, {} developed {} for {}",
        "The {} combined with {} to produce {}"
    ]
    
    # Word banks for diversity
    nouns = ["system", "technology", "process", "method", "approach", "technique", 
             "algorithm", "model", "framework", "structure", "network", "interface",
             "platform", "application", "software", "hardware", "device", "tool",
             "solution", "innovation", "discovery", "breakthrough", "advancement"]
    
    verbs = ["operates", "functions", "processes", "analyzes", "computes", "calculates",
             "synthesizes", "generates", "creates", "produces", "develops", "builds",
             "implements", "executes", "performs", "achieves", "accomplishes", "completes"]
    
    adjectives = ["efficient", "effective", "reliable", "accurate", "precise", "robust",
                  "scalable", "flexible", "adaptive", "intelligent", "sophisticated",
                  "advanced", "modern", "contemporary", "innovative", "revolutionary"]
    
    domains = ["artificial intelligence", "machine learning", "data science", "computing",
               "technology", "engineering", "research", "development", "innovation",
               "automation", "digital transformation", "information technology"]
    
    texts = []
    
    for i in range(num_texts):
        if i < 100:
            # Simple statements
            text = f"The {random.choice(nouns)} {random.choice(verbs)} {random.choice(adjectives)}ly in {random.choice(domains)}"
        elif i < 200:
            # Compound statements
            text = f"{random.choice(adjectives).title()} {random.choice(nouns)} and {random.choice(nouns)} {random.choice(verbs)} together"
        elif i < 300:
            # Technical descriptions
            text = f"Advanced {random.choice(nouns)} systems {random.choice(verbs)} complex data patterns with {random.choice(adjectives)} precision"
        elif i < 400:
            # Innovation focused
            text = f"Modern {random.choice(domains)} {random.choice(verbs)} new possibilities for {random.choice(adjectives)} solutions"
        else:
            # Use templates for variety
            template = random.choice(templates)
            text = template.format(
                random.choice(adjectives),
                random.choice(nouns),
                random.choice(domains),
                random.choice(verbs)
            )
        
        texts.append(text)
    
    print(f"✅ Generated {len(texts)} diverse text samples")
    return texts

def generate_ai_voices(texts):
    """Generate AI voices from text samples"""
    print("🎤 Generating AI voices...")
    
    current_count = get_current_ai_count()
    start_count = current_count + 1
    generated_count = 0
    
    for i, text in enumerate(texts):
        try:
            # Generate speech with slight variations
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save temporary file
            temp_file = os.path.join(OUTPUT_DIR, f'temp_{i}.mp3')
            tts.save(temp_file)
            
            # Load and convert to consistent format
            audio, sr = librosa.load(temp_file, sr=TARGET_SR)
            
            # Ensure correct length
            if len(audio) > TARGET_SAMPLES:
                audio = audio[:TARGET_SAMPLES]
            else:
                # Pad if shorter
                padding = TARGET_SAMPLES - len(audio)
                audio = np.pad(audio, (0, padding))
            
            # Save final file
            output_file = os.path.join(OUTPUT_DIR, f'ai_{start_count + i:03d}.wav')
            sf.write(output_file, audio, TARGET_SR)
            
            generated_count += 1
            
            # Progress and cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            # Progress update every 50 files
            if generated_count % 50 == 0:
                print(f"   ✅ Generated {generated_count}/{len(texts)} AI voices")
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Error generating voice {i}: {e}")
            continue
    
    return generated_count

def main():
    print("🚀 GENERATING 500 AI VOICES")
    print("=" * 50)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    current_count = get_current_ai_count()
    print(f"📊 Current AI voices: {current_count}")
    
    # Generate diverse texts
    texts = generate_diverse_texts(500)
    
    # Generate AI voices
    generated_count = generate_ai_voices(texts)
    
    # Final results
    final_count = get_current_ai_count()
    print(f"\n🎉 GENERATION COMPLETE!")
    print(f"📊 Generated {generated_count} new AI voices")
    print(f"📈 Total AI voices: {final_count}")
    print(f"⚖️  Dataset balance: 2000 real vs {final_count} AI")

if __name__ == "__main__":
    main()