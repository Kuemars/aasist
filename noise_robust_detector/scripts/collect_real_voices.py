"""
REAL VOICE COLLECTION SYSTEM - Methodical Approach
"""

import os
import sys

# Fix import path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
from project_config import *

class RealVoiceCollector:
    def __init__(self):
        self.output_dir = os.path.join(DATA_RAW, "clean_real")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def get_recording_script(self):
        """Return diverse text prompts for recording"""
        
        scripts = {
            "conversational": [
                "The quick brown fox jumps over the lazy dog",
                "Hello, my name is and I'm recording this for an AI research project",
                "I enjoy reading books and learning about new technologies",
                "The weather has been quite pleasant this time of year",
                "I think artificial intelligence will transform many industries"
            ],
            "technical": [
                "Machine learning models require large datasets for training",
                "Audio signal processing involves Fourier transforms and filtering techniques", 
                "Neural networks can learn complex patterns from data",
                "Digital signal processing is fundamental to modern audio systems",
                "Feature extraction is crucial for audio classification tasks"
            ],
            "emotional": [
                "I'm really excited about this project and its potential impact",
                "This is quite challenging but also very rewarding work", 
                "I feel optimistic about the future of voice technology",
                "It's fascinating how machines can understand human speech",
                "I'm curious to see how this detection system will perform"
            ],
            "storytelling": [
                "Once upon a time, there was a researcher studying voice patterns",
                "In a world filled with synthetic media, detection became crucial",
                "The journey to build robust AI systems was long but educational",
                "Through trial and error, the model gradually improved its accuracy",
                "Each recording added another piece to the complex puzzle"
            ]
        }
        return scripts
    
    def check_recording_environment(self):
        """Check if recording environment is suitable"""
        
        print("🔍 Checking recording environment...")
        
        # Test recording quality
        try:
            test_duration = 2  # seconds
            print(f"Recording {test_duration} second test sample...")
            
            test_audio = sd.rec(int(test_duration * SAMPLE_RATE), 
                              samplerate=SAMPLE_RATE, channels=1)
            sd.wait()
            
            # Analyze recording quality
            rms = np.sqrt(np.mean(test_audio**2))
            if rms < 0.01:
                print("❌ Recording level too low - check microphone")
                return False
            elif rms > 0.5:
                print("❌ Recording level too high - reduce input volume") 
                return False
            else:
                print(f"✅ Good recording level: {rms:.3f}")
                return True
                
        except Exception as e:
            print(f"❌ Recording test failed: {e}")
            return False
    
    def record_voice_sample(self, text, category, sample_num):
        """Record a single voice sample with metadata"""
        
        filename = f"real_{category}_{sample_num:03d}.wav"
        filepath = os.path.join(self.output_dir, filename)
        
        print(f"\n🎙️ Recording: '{text}'")
        print("   Press Enter to start recording (4 seconds)...")
        input()
        
        # Record audio
        audio = sd.rec(int(4 * SAMPLE_RATE), 
                      samplerate=SAMPLE_RATE, channels=1)
        print("   Recording...")
        sd.wait()
        
        # Save with metadata
        sf.write(filepath, audio, SAMPLE_RATE)
        
        # Verify recording
        try:
            verify_audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
            duration = len(verify_audio) / sr
            print(f"   ✅ Saved: {filename} ({duration:.1f}s)")
            return True
        except:
            print(f"   ❌ Failed to save: {filename}")
            return False
    
    def guided_collection_session(self):
        """Guide user through a complete recording session"""
        
        print("🎯 REAL VOICE COLLECTION SESSION")
        print("=" * 50)
        
        # Environment check
        if not self.check_recording_environment():
            print("Please fix recording issues before continuing.")
            return
        
        scripts = self.get_recording_script()
        total_recorded = 0
        
        print("\nWe'll record 4 samples from each of 4 categories (16 total)")
        print("This should take about 10-15 minutes")
        
        for category, texts in scripts.items():
            print(f"\n📝 Category: {category.upper()}")
            print("-" * 30)
            
            for i, text in enumerate(texts[:4]):  # First 4 from each category
                success = self.record_voice_sample(text, category, i + 1)
                if success:
                    total_recorded += 1
        
        print(f"\n🎉 Session complete! Recorded {total_recorded} samples")
        print(f"📍 Location: {self.output_dir}")

def main():
    collector = RealVoiceCollector()
    collector.guided_collection_session()

if __name__ == "__main__":
    main()
