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

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from models.RawNet2Spoof import Model

# Try to import pitch detector
try:
    from pitch_detector import load_pitch_classifier, predict_with_pitch
    PITCH_CLF_AVAILABLE = True
except ImportError:
    PITCH_CLF_AVAILABLE = False

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
    
    # Check for main model
    model_path = "models/AI_detector_Model_v1.pth"
    if not os.path.exists(model_path):
        print(f"❌ Main model not found at {model_path}")
        print("Looking for other models...")
        model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
        if model_files:
            model_path = f"models/{model_files[0]}"
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
    print(f"📥 Loading neural network from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Neural network loaded")
    
    # Load pitch classifier if available
    if PITCH_CLF_AVAILABLE:
        pitch_clf = load_pitch_classifier()
        if pitch_clf:
            print(f"✅ Pitch classifier loaded (84.5% accuracy)")
        else:
            print(f"⚠️  Pitch classifier not available")
            pitch_clf = None
    else:
        print(f"⚠️  Pitch detector module not installed")
        pitch_clf = None
    
    # Get test files
    test_files = []
    if os.path.exists(TEST_FOLDER):
        test_files = [os.path.join(TEST_FOLDER, f) for f in os.listdir(TEST_FOLDER) 
                     if f.endswith('.wav') or f.endswith('.mp3')]
    
    # Add verification samples
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
            # Preprocess for neural network
            audio_tensor = load_and_preprocess_audio(file_path).to(device)
            
            # Neural network prediction
            with torch.no_grad():
                output = model(audio_tensor)
                log_probs = output[1]  # Get logsoftmax output [batch, 2]
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
                pitch_pred, pitch_conf, pitch_feats = predict_with_pitch(file_path, pitch_clf)
                if pitch_pred is not None:
                    pitch_prediction = "HUMAN" if pitch_pred == 1 else "AI"
                    pitch_confidence = pitch_conf * 100  # Convert to percentage
                    
                    # Analyze pitch features
                    if pitch_feats is not None:
                        if pitch_feats[0] > 0.15:  # High log variance
                            pitch_analysis += "High pitch variance. "
                        if pitch_feats[2] < 80:  # Very low mean pitch
                            pitch_analysis += "Unusually low pitch. "
                        if pitch_feats[5] > 0.9:  # High voiced ratio
                            pitch_analysis += "High voiced ratio. "
            
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
            
            if pitch_prediction != "N/A":
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
            pitch_with_data = [r for r in results if r['pitch_prediction'] != 'N/A']
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
        agreements = sum(1 for r in results if r['pitch_prediction'] != 'N/A' 
                        and r['nn_prediction'] == r['pitch_prediction'])
        total_comparable = sum(1 for r in results if r['pitch_prediction'] != 'N/A')
        
        if total_comparable > 0:
            agreement_rate = 100 * agreements / total_comparable
            print(f"\n🤝 AGREEMENT RATE (Neural Net vs Pitch): {agreement_rate:.1f}%")
            print(f"   Agree: {agreements}/{total_comparable}")
    
    # Save results
    if results:
        with open('hybrid_detection_results.csv', 'w', newline='') as f:
            fieldnames = ['file', 'type', 'nn_prediction', 'nn_confidence', 
                         'nn_human_prob', 'nn_ai_prob', 'pitch_prediction', 
                         'pitch_confidence', 'pitch_analysis', 'ground_truth', 'correct']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n💾 Results saved to 'hybrid_detection_results.csv'")
    
    print("\n" + "=" * 60)
    print("ℹ️  SYSTEM COMPONENTS:")
    print("   1. Neural Network: 96.5% accuracy, learns complex patterns")
    print("   2. Pitch Classifier: 84.5% accuracy, analyzes pitch characteristics")
    print("   3. Hybrid system: Combines both for robust detection")
    print("=" * 60)

if __name__ == "__main__":
    detect_voices()
