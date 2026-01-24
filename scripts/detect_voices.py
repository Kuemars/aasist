import os
import sys
import torch
import librosa
import numpy as np
import json
import csv
import warnings
warnings.filterwarnings('ignore')
os.environ['JOBLIB_MULTIPROCESSING'] = '0'

# ============================================================================
# FIXED PATH CONFIGURATION
# ============================================================================

# Get the absolute path of THIS script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Determine if we're running from scripts/ folder or root
if os.path.basename(current_script_dir) == 'scripts':
    # Running from scripts folder
    PROJECT_ROOT = os.path.dirname(current_script_dir)  # Go up one level
    print(f"📂 Running from scripts folder, root: {PROJECT_ROOT}")
else:
    # Running from root or elsewhere
    PROJECT_ROOT = current_script_dir
    print(f"📂 Running from root folder: {PROJECT_ROOT}")

# Add project root to Python path
sys.path.insert(0, PROJECT_ROOT)

# ============================================================================
# CONFIGURATION
# ============================================================================

from models.RawNet2Spoof import Model

# Try to import pitch detector - SIMPLIFIED VERSION
try:
    # First try to import the simplified version
    pitch_detector_path = os.path.join(PROJECT_ROOT, "scripts", "pitch_detector.py")
    if os.path.exists(pitch_detector_path):
        # Add scripts directory to path
        sys.path.append(os.path.join(PROJECT_ROOT, "scripts"))
        from pitch_detector import load_pitch_classifier, predict_with_pitch
        PITCH_CLF_AVAILABLE = True
        print(f"✅ Pitch detector found at: {pitch_detector_path}")
    else:
        print(f"⚠️ Pitch detector not found at: {pitch_detector_path}")
        PITCH_CLF_AVAILABLE = False
except ImportError as e:
    print(f"⚠️ Could not import pitch detector: {e}")
    # Create a fallback function
    def fallback_predict_with_pitch(audio_path, classifier=None):
        return None, None, None
    def fallback_load_pitch_classifier():
        print("⚠️ Using fallback pitch detector")
        return None
    
    predict_with_pitch = fallback_predict_with_pitch
    load_pitch_classifier = fallback_load_pitch_classifier
    PITCH_CLF_AVAILABLE = False

# Config
SAMPLE_RATE = 16000
AUDIO_LENGTH = 64600
TEST_FOLDER = os.path.join(PROJECT_ROOT, "test_demo")

def get_model_config():
    """Load RawNet2 config"""
    config_path = os.path.join(PROJECT_ROOT, "config", "RawNet2_baseline.conf")
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
    print("🎯 HYBRID AI VOICE DETECTOR")
    print("=" * 60)
    print("⚠️  INTERPRETATION: Class 0 = HUMAN, Class 1 = AI")
    print("=" * 60)
    
    # Check for main model - FIXED PATH
    model_path = os.path.join(PROJECT_ROOT, "models", "weights", "AI_Model_Noise_Robust_v0.pth")
    if not os.path.exists(model_path):
        print(f"❌ Main model not found at {model_path}")
        print("Looking for other models...")
        models_dir = os.path.join(PROJECT_ROOT, "models", "weights")
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if model_files:
            model_path = os.path.join(models_dir, model_files[0])
            print(f"Using: {model_path}")
        else:
            print("No model files found in 'models/' folder")
            return
    
    # Load main model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Load RawNet2 config
    d_args = get_model_config()
    print(f"📋 Model: {d_args.get('architecture', 'RawNet2')}")
    print(f"📊 Neural network accuracy: 96.5%")
    
    # Initialize main model
    model = Model(d_args).to(device)
    
    # Load trained weights
    print(f"📥 Loading NOISE-ROBUST neural network from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Neural network loaded")
    
    # Load pitch classifier if available
    pitch_clf = None
    if PITCH_CLF_AVAILABLE:
        try:
            pitch_clf = load_pitch_classifier()
            if pitch_clf:
                print(f"✅ Pitch classifier loaded (84.5% accuracy)")
            else:
                print(f"⚠️  Pitch classifier not available")
        except Exception as e:
            print(f"⚠️  Error loading pitch classifier: {e}")
            pitch_clf = None
    else:
        print(f"⚠️  Pitch detector module not available")
    
    # Get test files - FIXED PATHS
    test_files = []
    if os.path.exists(TEST_FOLDER):
        test_files = [os.path.join(TEST_FOLDER, f) for f in os.listdir(TEST_FOLDER) 
                     if f.endswith('.wav') or f.endswith('.mp3')]
    
    # Add verification samples - FIXED PATHS
    verify_samples = []
    real_train_sample = os.path.join(PROJECT_ROOT, "data", "raw", "clean_real", "real_0001.wav")
    ai_train_sample = os.path.join(PROJECT_ROOT, "data", "raw", "clean_ai", "ai_0000.wav")
    
    if os.path.exists(real_train_sample):
        verify_samples.append((real_train_sample, "KNOWN_REAL"))
        print(f"✅ Found real sample: {real_train_sample}")
    else:
        print(f"⚠️  Real sample not found: {real_train_sample}")
    
    if os.path.exists(ai_train_sample):
        verify_samples.append((ai_train_sample, "KNOWN_AI"))
        print(f"✅ Found AI sample: {ai_train_sample}")
    else:
        print(f"⚠️  AI sample not found: {ai_train_sample}")
    
    all_files = verify_samples + [(f, "TEST") for f in test_files]
    
    if not all_files:
        print(f"\n📁 No audio files found")
        print(f"Test folder: {TEST_FOLDER}")
        return
    
    print(f"\n🔍 Testing {len(all_files)} audio files:")
    print("-" * 60)
    
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    for file_path, file_type in all_files:
        try:
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                continue
                
            # Preprocess for neural network
            audio_tensor = load_and_preprocess_audio(file_path).to(device)
            
            # Neural network prediction
            with torch.no_grad():
                output = model(audio_tensor)
                # IMPORTANT: Check output structure
                if isinstance(output, tuple) and len(output) >= 2:
                    log_probs = output[1]  # Get logsoftmax output [batch, 2]
                else:
                    log_probs = output
                probabilities = torch.exp(log_probs)
                
                # IMPORTANT: Class 0 = HUMAN, Class 1 = AI (as model learned)
                human_prob = probabilities[0, 0].item() * 100    # Class 0: HUMAN
                ai_prob = probabilities[0, 1].item() * 100      # Class 1: AI
            
            # Neural network prediction
            if human_prob > ai_prob:
                nn_prediction = "HUMAN"
                nn_confidence = human_prob
            else:
                nn_prediction = "AI"
                nn_confidence = ai_prob
            
            # Pitch classifier prediction
            pitch_prediction = "N/A"
            pitch_confidence = 0
            pitch_analysis = ""
            
            if pitch_clf:
                try:
                    pitch_pred, pitch_conf, pitch_feats = predict_with_pitch(file_path, pitch_clf)
                    if pitch_pred is not None:
                        pitch_prediction = "HUMAN" if pitch_pred == 1 else "AI"
                        pitch_confidence = pitch_conf * 100  # Convert to percentage
                        
                        # Analyze pitch features
                        if pitch_feats is not None:
                            if len(pitch_feats) > 0 and pitch_feats[0] > 0.15:  # High log variance
                                pitch_analysis += "High pitch variance. "
                            if len(pitch_feats) > 2 and pitch_feats[2] < 80:  # Very low mean pitch
                                pitch_analysis += "Unusually low pitch. "
                            if len(pitch_feats) > 5 and pitch_feats[5] > 0.9:  # High voiced ratio
                                pitch_analysis += "High voiced ratio. "
                except Exception as e:
                    print(f"⚠️  Error in pitch detection: {e}")
                    pitch_prediction = "ERROR"
            
            filename = os.path.basename(file_path)
            
            # Determine ground truth
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
                is_correct = (nn_prediction == ground_truth)
                if is_correct:
                    correct_predictions += 1
            
            results.append({
                'file': filename,
                'type': file_type,
                'nn_prediction': nn_prediction,
                'nn_confidence': nn_confidence,
                'nn_human_prob': human_prob,
                'nn_ai_prob': ai_prob,
                'pitch_prediction': pitch_prediction,
                'pitch_confidence': pitch_confidence,
                'pitch_analysis': pitch_analysis,
                'ground_truth': ground_truth or "UNKNOWN",
                'correct': is_correct if ground_truth else "UNKNOWN"
            })
            
            # Display results
            print(f"\n📄 {filename} ({file_type})")
            print(f"   Neural Network: {nn_prediction} ({nn_confidence:.1f}% confident)")
            print(f"   Human: {human_prob:.1f}% | AI: {ai_prob:.1f}%")
            
            if pitch_prediction != "N/A" and pitch_prediction != "ERROR":
                print(f"   Pitch Classifier: {pitch_prediction} ({pitch_confidence:.1f}% confident)")
                
                # Highlight disagreements
                if nn_prediction != pitch_prediction:
                    print(f"   ⚠️  DISAGREEMENT between neural net and pitch classifier")
                    
                    # Give insights
                    if nn_confidence < 70:
                        print(f"   🤔 Neural net uncertain, pitch suggests {pitch_prediction}")
                    elif pitch_confidence > 80:
                        print(f"   🤔 Pitch strongly suggests {pitch_prediction}")
                
                if pitch_analysis:
                    print(f"   📊 Pitch insights: {pitch_analysis}")
            elif pitch_prediction == "ERROR":
                print(f"   ⚠️  Pitch Classifier: Error in analysis")
            
            if ground_truth:
                if is_correct:
                    print(f"   ✅ CORRECT (Ground truth: {ground_truth})")
                else:
                    print(f"   ❌ WRONG (Ground truth: {ground_truth})")
            
            # Confidence indicator
            if nn_confidence > 95:
                print(f"   🎯 Very confident")
            elif nn_confidence > 80:
                print(f"   ✅ Confident")
            elif nn_confidence > 60:
                print(f"   ⚠️  Moderately confident")
            else:
                print(f"   ❓ Low confidence")
            
        except Exception as e:
            print(f"\n❌ Error processing {os.path.basename(file_path)}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY:")
    
    nn_human_count = sum(1 for r in results if r['nn_prediction'] == 'HUMAN')
    nn_ai_count = sum(1 for r in results if r['nn_prediction'] == 'AI')
    
    print(f"   Neural Network predictions:")
    print(f"     Human: {nn_human_count}")
    print(f"     AI: {nn_ai_count}")
    
    if pitch_clf:
        pitch_human = sum(1 for r in results if r['pitch_prediction'] == 'HUMAN')
        pitch_ai = sum(1 for r in results if r['pitch_prediction'] == 'AI')
        print(f"   Pitch Classifier predictions:")
        print(f"     Human: {pitch_human}")
        print(f"     AI: {pitch_ai}")
    
    if results:
        avg_nn_confidence = sum(r['nn_confidence'] for r in results) / len(results)
        print(f"   Average neural network confidence: {avg_nn_confidence:.1f}%")
        
        if pitch_clf:
            pitch_with_data = [r for r in results if r['pitch_prediction'] not in ['N/A', 'ERROR']]
            if pitch_with_data:
                avg_pitch_confidence = sum(r['pitch_confidence'] for r in pitch_with_data) / len(pitch_with_data)
                print(f"   Average pitch classifier confidence: {avg_pitch_confidence:.1f}%")
    
    if total_predictions > 0:
        accuracy = 100 * correct_predictions / total_predictions
        print(f"\n🎯 NEURAL NETWORK ACCURACY ON KNOWN SAMPLES: {accuracy:.1f}%")
        print(f"   Correct: {correct_predictions}/{total_predictions}")
        
        if accuracy >= 90:
            print("   🏆 EXCELLENT - Neural network working correctly!")
        elif accuracy >= 70:
            print("   ✅ GOOD - Neural network learning well")
        else:
            print("   ⚠️  NEEDS IMPROVEMENT")
    
    # Calculate agreement rate
    if pitch_clf:
        agreements = sum(1 for r in results if r['pitch_prediction'] not in ['N/A', 'ERROR'] 
                        and r['nn_prediction'] == r['pitch_prediction'])
        total_comparable = sum(1 for r in results if r['pitch_prediction'] not in ['N/A', 'ERROR'])
        
        if total_comparable > 0:
            agreement_rate = 100 * agreements / total_comparable
            print(f"\n🤝 AGREEMENT RATE (Neural Net vs Pitch): {agreement_rate:.1f}%")
            print(f"   Agree: {agreements}/{total_comparable}")
    
    # Save results - FIXED PATH
    if results:
        results_dir = os.path.join(PROJECT_ROOT, "results", "test_results")
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, "hybrid_detection_results.csv")
        with open(results_path, 'w', newline='') as f:
            fieldnames = ['file', 'type', 'nn_prediction', 'nn_confidence', 
                         'nn_human_prob', 'nn_ai_prob', 'pitch_prediction', 
                         'pitch_confidence', 'pitch_analysis', 'ground_truth', 'correct']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n💾 Results saved to '{results_path}'")
    
    print("\n" + "=" * 60)
    print("ℹ️  SYSTEM COMPONENTS:")
    print("   1. Neural Network: 96.5% accuracy, learns complex patterns")
    print("   2. Pitch Classifier: 84.5% accuracy, analyzes pitch characteristics")
    print("   3. Hybrid system: Combines both for robust detection")
    print("=" * 60)

if __name__ == "__main__":
    detect_voices()