import os
import sys
import time
import numpy as np

# Add the parent directory to path to import project_config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from gtts import gTTS
import soundfile as sf
import librosa

# Import config directly
SAMPLE_RATE = 16000
AUDIO_LENGTH = 64600  # 4.04 seconds
DATA_RAW = "data/raw"
AI_RAW_PATH = os.path.join(DATA_RAW, "clean_ai")

def generate_ai_voices():
    """Generate additional AI voices using gTTS"""
    
    # Texts for diverse AI voice generation
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Artificial intelligence is transforming our world",
        "Machine learning models require diverse datasets",
        "Voice recognition systems are becoming more accurate",
        "Digital assistants help people daily with various tasks",
        "Neural networks can process complex audio patterns",
        "Technology continues to evolve at a rapid pace",
        "Computer science principles apply to many fields",
        "Data preprocessing is essential for good model performance",
        "Audio signal processing involves frequency analysis",
        "Deep learning has revolutionized many industries",
        "Natural language processing understands human language",
        "The future of technology holds many possibilities",
        "Software development requires careful planning and testing",
        "Algorithms can learn from data patterns effectively",
        "Cloud computing enables scalable applications",
        "Mobile devices have powerful computing capabilities",
        "User interfaces should be intuitive and accessible",
        "Security is crucial in modern software systems",
        "Open source projects benefit the developer community",
        "The internet connects people across the globe",
        "Digital transformation affects every industry",
        "Automation improves efficiency in many processes",
        "Quality assurance ensures reliable software performance",
        "Agile methodologies help teams adapt to changes",
        "Version control systems manage code collaboration",
        "Continuous integration streamlines development workflows",
        "Technical documentation helps maintain projects long term",
        "Problem solving skills are essential for developers",
        "Innovation drives progress in the tech industry",
        "Collaboration tools enable remote team coordination",
        "Data visualization makes complex information understandable",
        "Cybersecurity protects sensitive digital information",
        "Edge computing processes data closer to the source",
        "Quantum computing may solve previously intractable problems"
    ]
    
    # Create output directory
    os.makedirs(AI_RAW_PATH, exist_ok=True)
    
    # Get current AI files to continue numbering
    existing_files = [f for f in os.listdir(AI_RAW_PATH) if f.startswith('ai_') and f.endswith('.wav')]
    start_idx = len(existing_files)
    
    print(f"Found {start_idx} existing AI voices. Generating {len(texts)} more...")
    
    for i, text in enumerate(texts):
        try:
            # Generate speech with slight variations
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save temporary file
            temp_file = os.path.join(AI_RAW_PATH, f'temp_{i}.mp3')
            tts.save(temp_file)
            
            # Load and convert to consistent format
            audio, sr = librosa.load(temp_file, sr=SAMPLE_RATE)
            
            # Ensure correct length
            if len(audio) > AUDIO_LENGTH:
                audio = audio[:AUDIO_LENGTH]
            else:
                # Pad if shorter
                padding = AUDIO_LENGTH - len(audio)
                audio = np.pad(audio, (0, padding))
            
            # Save final file
            output_file = os.path.join(AI_RAW_PATH, f'ai_{start_idx + i + 1:03d}.wav')
            sf.write(output_file, audio, SAMPLE_RATE)
            
            print(f"Generated AI voice {start_idx + i + 1:03d}: {text[:50]}...")
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            # Small delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error generating voice {i}: {e}")
            continue
    
    print(f"AI voice generation complete. Total AI voices: {start_idx + len(texts)}")

if __name__ == "__main__":
    generate_ai_voices()