"""
AI VOICE COLLECTION PLAN - Scale to Match Real Voices
"""

import os
import sys

# Fix import path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from project_config import *

def create_ai_voice_plan():
    """Plan for collecting 200+ diverse AI voices"""
    
    print("🎯 AI VOICE COLLECTION PLAN")
    print("=" * 50)
    
    print("📊 TARGET: 200+ AI voice samples (matching real voice count)")
    
    ai_sources = {
        "ElevenLabs": {
            "samples": 80,
            "quality": "Highest quality, most realistic",
            "diversity": "Multiple voices, emotions, styles",
            "method": "API or manual download from free tier",
            "cost": "Free tier available"
        },
        "gTTS (Google)": {
            "samples": 60, 
            "quality": "Good quality, robotic but clear",
            "diversity": "Multiple accents (US, UK, AU, CA, IE)",
            "method": "Python script automation",
            "cost": "Free"
        },
        "Azure TTS": {
            "samples": 40,
            "quality": "High quality, natural sounding",
            "diversity": "Neural voices, different styles",
            "method": "Free tier API",
            "cost": "Free credits available"
        },
        "Amazon Polly": {
            "samples": 20,
            "quality": "High quality",
            "diversity": "Different voice engines",
            "method": "Free tier",
            "cost": "Free tier available"
        }
    }
    
    total_planned = 0
    for source, details in ai_sources.items():
        print(f"\n🎯 {source}:")
        print(f"   Samples: {details['samples']}")
        print(f"   Quality: {details['quality']}")
        print(f"   Method: {details['method']}")
        total_planned += details['samples']
    
    print(f"\n📈 TOTAL PLANNED: {total_planned} AI voice samples")
    return ai_sources, total_planned

def create_execution_order():
    """Create step-by-step execution order"""
    
    print("\n📋 EXECUTION ORDER:")
    
    steps = [
        ("1. ElevenLabs Setup", "Sign up for free tier, download 80 samples", "1-2 hours"),
        ("2. gTTS Automation", "Run script to generate 60 samples", "30 minutes"),
        ("3. Azure TTS", "Use free credits for 40 samples", "1 hour"),
        ("4. Amazon Polly", "Optional - 20 samples for diversity", "30 minutes"),
        ("5. Verification", "Check all 200 AI samples", "30 minutes")
    ]
    
    for step, action, time in steps:
        print(f"   {step:20} {action:45} {time}")

def immediate_next_step():
    """What to do right now"""
    
    print("\n🎯 IMMEDIATE NEXT STEP:")
    print("   Start with ElevenLabs - highest quality AI voices")
    print("   Go to: https://elevenlabs.io")
    print("   Sign up for free account (10,000 characters/month)")
    print("   Download 80 diverse voice samples")
    print("   Save to: noise_robust_detector/data/raw/clean_ai/")

def main():
    print("🎯 STEP 6: AI VOICE COLLECTION PLANNING")
    print("=" * 60)
    
    # Create the plan
    ai_sources, total_planned = create_ai_voice_plan()
    
    # Execution order
    create_execution_order()
    
    # Immediate action
    immediate_next_step()
    
    print("\n" + "=" * 60)
    print("✅ PLAN READY: Collect 200+ diverse AI voices")
    print("💡 Start with ElevenLabs for highest quality")
    print("   This balances your 200 real voices")
    print("=" * 60)

if __name__ == "__main__":
    main()