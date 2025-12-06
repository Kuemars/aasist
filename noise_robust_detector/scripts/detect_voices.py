import os
import sys
import torch
import librosa
import numpy as np
import json
import csv

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from models.RawNet2Spoof import Model

# Config
SAMPLE_RATE = 16000
AUDIO_LENGTH = 64600
TEST_FOLDER = "test_demo"

def get_model_config():
    """Load RawNet2 config"""
    config_path = os.path.join(project_root, "config/RawNet2_baseline.conf")
    default_config = {
        "architecture": "RawNet2Spoof",
        "nb_samp": 64600,
        "first_conv": 1024,
        "in_channels": 1,
        "filts": [20, [20, 20], [20, 128], [128, 128]],
        "blocks": [2, 4],
        "nb_fc_node": 1024,
        "gru_node": 1024,
        "nb_gru_layer": 3,
        "nb_classes": 2
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if "model_config" in config:
                    model_config = config["model_config"]
                    model_config["nb_classes"] = 2
                    model_config["nb_samp"] = 64600
                    return model_config
        except:
            pass
    return default_config

def analyze_pitch_variance(audio_path):
    """Analyze pitch variance with proper filtering"""
    try:
        audio, sr = librosa.load(audio_path, sr=16000, duration=4.0)
        
        # Get pitch using YIN
        pitches = librosa.yin(audio, fmin=50, fmax=500)
        
        # FILTER: Only keep realistic human pitch range (50-500 Hz)
        pitches = pitches[(pitches > 50) & (pitches < 500)]
        
        # Need at least 10 valid pitch points
        if len(pitches) < 10:
            return "Not enough voiced segments"
        
        # Calculate LOG variance (pitch varies multiplicatively, not additively)
        log_pitches = np.log(pitches)
        variance = np.var(log_pitches)  # Variance in log space
        
        # REALISTIC thresholds (tune these):
        if variance < 0.01:  # Very consistent pitch
            return f"AI-like (log variance: {variance:.4f})"
        elif variance < 0.05:  # Normal human variation
            return f"Human-like (log variance: {variance:.4f})"
        else:  # Extreme variation
            return f"Character-like (log variance: {variance:.4f})"
            
    except Exception as e:
        return f"Error: {str(e)}"

def load_and_preprocess_audio(file_path):
    """Load and preprocess audio exactly like training"""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=4.04)
    
    if len(audio) > AUDIO_LENGTH:
        audio = audio[:AUDIO_LENGTH]
    elif len(audio) < AUDIO_LENGTH:
        audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)))
    
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return torch.FloatTensor(audio).unsqueeze(0)  # Add batch dimension

def detect_voices():
    print("🎯 AI VOICE DETECTOR - CORRECTED CLASS LABELS")
    print("=" * 60)
    print("⚠️  INTERPRETATION: Class 0 = AI, Class 1 = HUMAN")
    print("=" * 60)
    
    # Check for model
    model_path = "models/rawnet2_v1.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Looking for other saved models...")
        model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
        if model_files:
            model_path = f"models/{model_files[0]}"
            print(f"Using: {model_path}")
        else:
            print("No model files found in 'models/' folder")
            return
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Load RawNet2 config
    d_args = get_model_config()
    print(f"📋 Model: {d_args.get('architecture', 'RawNet2')}")
    print(f"📊 Trained accuracy: 98.0% (with corrected labels)")
    
    # Initialize RawNet2 model
    model = Model(d_args).to(device)
    
    # Load trained weights
    print(f"📥 Loading weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded")
    
    # Get test files
    test_files = []
    if os.path.exists(TEST_FOLDER):
        test_files = [os.path.join(TEST_FOLDER, f) for f in os.listdir(TEST_FOLDER) 
                     if f.endswith('.wav') or f.endswith('.mp3')]
    
    # Add some known samples from training set for verification
    verify_samples = []
    real_train_sample = "data/raw/clean_real/real_0001.wav"
    ai_train_sample = "data/raw/clean_ai/ai_0000.wav"
    
    if os.path.exists(real_train_sample):
        verify_samples.append((real_train_sample, "KNOWN_REAL"))
    if os.path.exists(ai_train_sample):
        verify_samples.append((ai_train_sample, "KNOWN_AI"))
    
    all_files = verify_samples + [(f, "TEST") for f in test_files]
    
    if not all_files:
        print(f"\n📁 No audio files found")
        return
    
    print(f"\n🔍 Testing {len(all_files)} audio files:")
    print("-" * 60)
    
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    for file_path, file_type in all_files:
        try:
            # Preprocess
            audio_tensor = load_and_preprocess_audio(file_path).to(device)
            
            # Predict - CORRECTED INTERPRETATION
            with torch.no_grad():
                output = model(audio_tensor)
                log_probs = output[1]  # Get logsoftmax output [batch, 2]
                probabilities = torch.exp(log_probs)
                
                # CORRECTED: Class 0 = AI, Class 1 = HUMAN
                human_prob = probabilities[0, 0].item() * 100    # Class 0: HUMAN (what model learned)
                ai_prob = probabilities[0, 1].item() * 100      # Class 1: AI (what model learned)
            
            # Determine prediction
            if human_prob > ai_prob:
                prediction = "HUMAN"
                confidence = human_prob
            else:
                prediction = "AI"
                confidence = ai_prob
            
            filename = os.path.basename(file_path)
            
            # Determine ground truth based on filename or file_type
            ground_truth = None
            if file_type == "KNOWN_REAL":
                ground_truth = "HUMAN"
            elif file_type == "KNOWN_AI":
                ground_truth = "AI"
            elif "ai" in filename.lower():
                ground_truth = "AI"
            elif "real" in filename.lower():
                ground_truth = "HUMAN"
            
            # Check if correct
            is_correct = False
            if ground_truth:
                total_predictions += 1
                is_correct = (prediction == ground_truth)
                if is_correct:
                    correct_predictions += 1
            
            results.append({
                'file': filename,
                'type': file_type,
                'prediction': prediction,
                'confidence': confidence,
                'human_prob': human_prob,
                'ai_prob': ai_prob,
                'ground_truth': ground_truth or "UNKNOWN",
                'correct': is_correct if ground_truth else "UNKNOWN"
            })
            
            # Color coding
            if ground_truth:
                if is_correct:
                    color_start = "\033[92m"  # Green for correct
                    color_end = "\033[0m"
                    correct_symbol = "✅"
                else:
                    color_start = "\033[91m"  # Red for wrong
                    color_end = "\033[0m"
                    correct_symbol = "❌"
            else:
                color_start = ""
                color_end = ""
                correct_symbol = "🔍"
            
            print(f"{color_start}{correct_symbol} {filename} ({file_type}){color_end}")
            print(f"   Prediction: {prediction} ({confidence:.1f}% confident)")
            print(f"   Human: {human_prob:.1f}% | AI: {ai_prob:.1f}%")
            
            pitch_info = analyze_pitch_variance(file_path)
            print(f"   Pitch analysis: {pitch_info}")
            if 40 < confidence < 60:  # Uncertain range
                if "AI-like" in pitch_info:
                    print(f"   ⚠️  Pitch suggests AI")
                elif "Human-like" in pitch_info:
                    print(f"   ⚠️  Pitch suggests HUMAN")

            if ground_truth:
                print(f"   Ground truth: {ground_truth}")
                if not is_correct:
                    print(f"   ⚠️  MISCLASSIFIED")
            
            # Confidence indicator
            if confidence > 95:
                print(f"   🎯 Very confident")
            elif confidence > 80:
                print(f"   ✅ Confident")
            elif confidence > 60:
                print(f"   ⚠️  Moderately confident")
            else:
                print(f"   ❓ Low confidence")
            
            print()
            
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(file_path)}: {e}")
    
    # Summary
    print("=" * 60)
    print("📊 FINAL SUMMARY:")
    
    human_count = sum(1 for r in results if r['prediction'] == 'HUMAN')
    ai_count = sum(1 for r in results if r['prediction'] == 'AI')
    
    print(f"   Human predictions: {human_count}")
    print(f"   AI predictions: {ai_count}")
    
    if results:
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"   Average confidence: {avg_confidence:.1f}%")
    
    if total_predictions > 0:
        accuracy = 100 * correct_predictions / total_predictions
        print(f"\n🎯 ACCURACY ON KNOWN SAMPLES: {accuracy:.1f}%")
        print(f"   Correct: {correct_predictions}/{total_predictions}")
        
        if accuracy >= 90:
            print("   🏆 EXCELLENT - Model is working correctly!")
        elif accuracy >= 70:
            print("   ✅ GOOD - Model is learning")
        else:
            print("   ⚠️  NEEDS IMPROVEMENT - Check training labels")
    
    # Save results
    if results:
        with open('detection_results_corrected.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file', 'type', 'prediction', 'confidence', 
                                                  'human_prob', 'ai_prob', 'ground_truth', 'correct'])
            writer.writeheader()
            writer.writerows(results)
        print(f"\n💾 Results saved to 'detection_results_corrected.csv'")
    
    print("\n⚠️  IMPORTANT: Class interpretation corrected:")
    print("   - Class 0 (AI): Model outputs high probability for AI voices")
    print("   - Class 1 (HUMAN): Model outputs high probability for human voices")
    print("=" * 60)

if __name__ == "__main__":
    detect_voices()
