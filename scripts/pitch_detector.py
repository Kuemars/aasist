import os
import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')

# Get the absolute path of THIS script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Determine if we're running from scripts/ folder or root
if os.path.basename(current_script_dir) == 'scripts':
    # Running from scripts folder
    PROJECT_ROOT = os.path.dirname(current_script_dir)
else:
    # Running from root or elsewhere
    PROJECT_ROOT = current_script_dir

def extract_pitch_features(audio_path):
    """
    Extract 6 pitch-based features from audio file.
    Returns: numpy array of 6 features or None if extraction fails.
    """
    try:
        # Load audio (first 4 seconds)
        audio, sr = librosa.load(audio_path, sr=16000, duration=4.0)
        
        # Get pitch contour using YIN algorithm
        pitches = librosa.yin(audio, fmin=50, fmax=500)
        
        # Filter invalid values (silence, errors)
        valid_pitches = pitches[(pitches > 50) & (pitches < 500)]
        
        # Need minimum data points
        if len(valid_pitches) < 10:
            return None
        
        # 1. Log variance (pitch stability)
        log_variance = np.var(np.log(valid_pitches + 1e-8))
        
        # 2. Pitch range (Hz)
        pitch_range = np.max(valid_pitches) - np.min(valid_pitches)
        
        # 3. Mean pitch (Hz)
        mean_pitch = np.mean(valid_pitches)
        
        # 4. Average pitch jumps
        pitch_jumps = np.mean(np.abs(np.diff(valid_pitches)))
        
        # 5. Pitch smoothness (inverse of variance of differences)
        diff_variance = np.var(np.diff(valid_pitches))
        pitch_smoothness = 1.0 / (1.0 + diff_variance)
        
        # 6. Voiced ratio (percentage of audio with detectable pitch)
        voiced_ratio = len(valid_pitches) / len(pitches)
        
        return np.array([log_variance, pitch_range, mean_pitch, 
                        pitch_jumps, pitch_smoothness, voiced_ratio])
        
    except Exception as e:
        print(f"⚠️ Pitch extraction failed: {e}")
        return None

def simple_pitch_prediction(features):
    """
    Improved rule-based pitch prediction based on known characteristics:
    - AI voices often have unnaturally stable pitch (low variance)
    - AI voices often have restricted pitch range
    - AI voices may have unnatural pitch jumps
    - Human voices have more natural variation
    
    Returns: 0=AI, 1=Human, confidence score
    """
    if features is None:
        return None, 0
    
    log_var, pitch_range, mean_pitch, jumps, smoothness, voiced_ratio = features
    
    # Initialize score (positive = human, negative = AI)
    score = 0
    
    # Rule 1: Log Variance
    # Very low variance (<0.05) is often AI (too stable)
    # Medium variance (0.05-0.15) is human-like
    # Very high variance (>0.2) is also human (emotional speech)
    if log_var < 0.05:
        score -= 2  # AI - unnaturally stable
    elif log_var < 0.15:
        score += 1  # Human - natural variation
    elif log_var < 0.25:
        score += 2  # Human - expressive
    else:
        score += 1  # Very high variance is human
    
    # Rule 2: Pitch Range
    # Very narrow range (<100 Hz) is AI-like
    # Wide range (>200 Hz) is human-like
    if pitch_range < 100:
        score -= 1
    elif pitch_range > 200:
        score += 2
    elif pitch_range > 300:
        score += 3  # Very expressive human
    
    # Rule 3: Mean Pitch
    # Check if within natural human ranges
    # Male: 85-180 Hz, Female: 165-255 Hz
    if (mean_pitch >= 85 and mean_pitch <= 180) or (mean_pitch >= 165 and mean_pitch <= 255):
        score += 1  # Natural pitch
    elif mean_pitch < 50 or mean_pitch > 400:
        score -= 1  # Unnatural pitch
    
    # Rule 4: Pitch Jumps
    # Very large jumps (>80 Hz) are unnatural
    # Moderate jumps (20-60 Hz) are normal
    if jumps > 80:
        score -= 2
    elif jumps > 40:
        score -= 1
    elif jumps > 15:
        score += 1  # Natural human variation
    
    # Rule 5: Pitch Smoothness
    # Very high smoothness (>0.7) is AI-like (too perfect)
    # Low smoothness (<0.3) is human-like
    if smoothness > 0.7:
        score -= 1
    elif smoothness < 0.3:
        score += 1
    
    # Rule 6: Voiced Ratio
    # Very high ratio (>0.95) can be AI (continuous speech)
    # Normal human speech has some unvoiced segments
    if voiced_ratio > 0.95:
        score -= 1
    elif voiced_ratio > 0.7:
        score += 1  # Normal
    
    # Convert score to prediction
    if score >= 0:
        prediction = 1  # Human
        confidence = min((score + 5) / 15, 1.0)  # Scale to 0-1
    else:
        prediction = 0  # AI
        confidence = min(abs(score) / 10, 1.0)
    
    # Ensure confidence is reasonable
    confidence = max(0.1, min(confidence, 0.9))
    
    return prediction, confidence

def load_pitch_classifier():
    """Simple pitch classifier - always returns the rule-based predictor"""
    print("✅ Loaded IMPROVED rule-based pitch classifier")
    return "rule_based"  # Marker for rule-based system

def predict_with_pitch(audio_path, classifier=None):
    """
    Predict Human/AI using pitch features.
    Returns: (prediction, confidence, features)
    prediction: 0=AI, 1=Human
    confidence: probability of prediction (0-1)
    features: array of 6 pitch features
    """
    # Extract features
    features = extract_pitch_features(audio_path)
    if features is None:
        return None, None, None
    
    # Use improved rule-based prediction
    prediction, confidence = simple_pitch_prediction(features)
    
    return prediction, confidence, features

def analyze_audio_with_pitch(audio_path):
    """Complete analysis of audio file using pitch features"""
    print(f"\n🔍 Pitch Analysis: {os.path.basename(audio_path)}")
    print("-" * 40)
    
    pred, conf, feats = predict_with_pitch(audio_path)
    
    if pred is not None:
        label = "HUMAN" if pred == 1 else "AI"
        print(f"Pitch-based prediction: {label}")
        print(f"Confidence: {conf:.1%}")
        
        # Show individual features with interpretation
        feature_names = ['Log Variance', 'Pitch Range', 'Mean Pitch', 
                        'Pitch Jumps', 'Pitch Smoothness', 'Voiced Ratio']
        
        interpretations = [
            f"{'AI' if feats[0] < 0.05 else 'HUMAN'} (<0.05=AI, >0.15=HUMAN)",
            f"{'HUMAN' if feats[1] > 200 else 'AI'} (>200=HUMAN, <100=AI)",
            f"{'HUMAN' if (85<=feats[2]<=180) or (165<=feats[2]<=255) else 'AI'}",
            f"{'AI' if feats[3] > 60 else 'HUMAN'} (>60=AI, <40=HUMAN)",
            f"{'AI' if feats[4] > 0.7 else 'HUMAN'} (>0.7=AI, <0.3=HUMAN)",
            f"{'AI' if feats[5] > 0.95 else 'HUMAN'} (>0.95=AI, 0.7-0.9=HUMAN)"
        ]
        
        print("\nPitch features with interpretation:")
        for i, (name, value) in enumerate(zip(feature_names, feats)):
            print(f"  {name:20} {value:.4f}  {interpretations[i]}")
        
        return pred, conf, feats
    else:
        print("❌ Could not extract pitch features")
        return None

if __name__ == "__main__":
    # Test with sample files
    print("🎵 PITCH DETECTOR - IMPROVED RULE BASED SYSTEM")
    print("=" * 60)
    print("Based on pitch characteristics:")
    print("  - AI: Too stable, narrow range, unnatural jumps")
    print("  - HUMAN: Natural variation, wider range, expressive")
    print("=" * 60)
    
    test_demo_dir = os.path.join(PROJECT_ROOT, "test_demo")
    test_files = []
    
    if os.path.exists(test_demo_dir):
        test_files = [os.path.join(test_demo_dir, f) for f in os.listdir(test_demo_dir) 
                     if f.endswith('.wav')]
    
    if test_files:
        for test_file in test_files:
            if os.path.exists(test_file):
                analyze_audio_with_pitch(test_file)
    else:
        print("No test files found in test_demo/")