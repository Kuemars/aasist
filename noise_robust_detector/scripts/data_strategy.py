"""
DATA COLLECTION STRATEGY - Comprehensive Plan
"""

import os
import sys

# Fix import path - add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from project_config import *

def analyze_data_requirements():
    """Analyze exactly what data we need and why"""
    
    print("📊 DATA REQUIREMENTS ANALYSIS")
    print("=" * 50)
    
    requirements = {
        "clean_real_voices": {
            "purpose": "Base real voice samples before noise addition",
            "sources": ["Your recordings", "LibriSpeech", "Common Voice"],
            "target_count": 100,
            "diversity_factors": ["gender", "age", "accent", "speaking_style"],
            "quality_standards": ["clear_audio", "no_background_noise", "consistent_levels"]
        },
        "clean_ai_voices": {
            "purpose": "Base AI voice samples before noise addition", 
            "sources": ["ElevenLabs", "gTTS", "Azure TTS", "Amazon Polly"],
            "target_count": 100,
            "diversity_factors": ["tts_provider", "voice_type", "speaking_rate"],
            "quality_standards": ["high_quality", "varied_content", "different_engines"]
        },
        "noise_samples": {
            "purpose": "Background noises to augment training data",
            "sources": ["ESC-50 dataset", "self_recorded", "online_sources"],
            "target_count": 20,
            "diversity_factors": ["environment", "intensity", "frequency_profile"],
            "quality_standards": ["clean_recordings", "varied_intensities", "realistic"]
        }
    }
    
    # Print requirements
    for category, specs in requirements.items():
        print(f"\n🎯 {category.upper().replace('_', ' ')}:")
        print(f"   Purpose: {specs['purpose']}")
        print(f"   Target: {specs['target_count']} samples")
        print(f"   Sources: {', '.join(specs['sources'])}")
        print(f"   Diversity: {', '.join(specs['diversity_factors'])}")
    
    return requirements

def create_collection_priority():
    """Create priority order for data collection"""
    
    print("\n🎯 COLLECTION PRIORITY ORDER:")
    priorities = [
        ("1. Clean Real Voices", "Foundation - get 50 diverse real voices first"),
        ("2. Clean AI Voices", "Foundation - get 50 diverse AI voices"), 
        ("3. Noise Samples", "Augmentation - collect 10 background noise types"),
        ("4. Expand Real Voices", "Scale to 100 real voices"),
        ("5. Expand AI Voices", "Scale to 100 AI voices"),
        ("6. Special Cases", "Edge cases and challenging samples")
    ]
    
    for priority, reason in priorities:
        print(f"   {priority}: {reason}")
    
    return priorities

def estimate_time_requirements():
    """Estimate time needed for each phase"""
    
    print("\n⏰ TIME ESTIMATES:")
    time_estimates = [
        ("Real Voice Recording", "2-3 hours", "50 samples @ 2-3 minutes each"),
        ("AI Voice Generation", "1-2 hours", "Automated, mostly waiting"),
        ("Noise Collection", "1 hour", "Download datasets + some recording"),
        ("Data Validation", "1 hour", "Quality checks and organization"),
        ("Total Data Phase", "5-7 hours", "Spread over 2-3 days")
    ]
    
    for task, time, notes in time_estimates:
        print(f"   {task}: {time} ({notes})")

def main():
    print("🎯 STEP 2: DATA COLLECTION STRATEGY")
    print("=" * 60)
    
    # Analyze requirements
    requirements = analyze_data_requirements()
    
    # Create priority order
    priorities = create_collection_priority()
    
    # Time estimates
    estimate_time_requirements()
    
    print("\n" + "=" * 60)
    print("✅ STRATEGY COMPLETE")
    print("💡 Next: Begin with Priority 1 - Clean Real Voices")
    print("=" * 60)

if __name__ == "__main__":
    main()