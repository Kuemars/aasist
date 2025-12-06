import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import datetime

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
        log_variance = np.var(np.log(valid_pitches))
        
        # 2. Pitch range (Hz)
        pitch_range = np.max(valid_pitches) - np.min(valid_pitches)
        
        # 3. Mean pitch (Hz)
        mean_pitch = np.mean(valid_pitches)
        
        # 4. Average pitch jumps
        pitch_jumps = np.mean(np.abs(np.diff(valid_pitches)))
        
        # 5. Pitch smoothness (inverse of variance of differences)
        pitch_smoothness = 1.0 / (1.0 + np.var(np.diff(valid_pitches)))
        
        # 6. Voiced ratio (percentage of audio with detectable pitch)
        voiced_ratio = len(valid_pitches) / len(pitches)
        
        return np.array([log_variance, pitch_range, mean_pitch, 
                        pitch_jumps, pitch_smoothness, voiced_ratio])
        
    except Exception as e:
        print(f"⚠️ Pitch extraction failed for {os.path.basename(audio_path)}: {e}")
        return None

def collect_all_training_data(real_dir="data/raw/clean_real", 
                             ai_dir="data/raw/clean_ai"):
    """
    Collect pitch features from ALL real and AI voice samples.
    Returns: X (features), y (labels: 0=AI, 1=Human)
    """
    X = []
    y = []
    
    print("📊 Collecting ALL REAL voice pitch features...")
    real_count = 0
    failed_real = 0
    
    if os.path.exists(real_dir):
        files = [f for f in os.listdir(real_dir) if f.endswith('.wav')]
        total_real = len(files)
        
        for idx, filename in enumerate(files, 1):
            filepath = os.path.join(real_dir, filename)
            features = extract_pitch_features(filepath)
            
            if features is not None:
                X.append(features)
                y.append(1)  # Label 1 = Human
                real_count += 1
            else:
                failed_real += 1
            
            # Show progress every 100 files
            if idx % 100 == 0 or idx == total_real:
                print(f"  Real: {idx}/{total_real} processed ({real_count} successful, {failed_real} failed)")
    
    print("\n📊 Collecting ALL AI voice pitch features...")
    ai_count = 0
    failed_ai = 0
    
    if os.path.exists(ai_dir):
        files = [f for f in os.listdir(ai_dir) if f.endswith('.wav')]
        total_ai = len(files)
        
        for idx, filename in enumerate(files, 1):
            filepath = os.path.join(ai_dir, filename)
            features = extract_pitch_features(filepath)
            
            if features is not None:
                X.append(features)
                y.append(0)  # Label 0 = AI
                ai_count += 1
            else:
                failed_ai += 1
            
            # Show progress every 100 files
            if idx % 100 == 0 or idx == total_ai:
                print(f"  AI: {idx}/{total_ai} processed ({ai_count} successful, {failed_ai} failed)")
    
    print(f"\n✅ Collection complete:")
    print(f"   Real voices: {real_count} (failed: {failed_real})")
    print(f"   AI voices: {ai_count} (failed: {failed_ai})")
    print(f"   Total successful: {len(X)} samples")
    
    if len(X) > 0:
        print(f"   Features per sample: {len(X[0])}")
    
    # Verify balance
    if real_count > 0 and ai_count > 0:
        balance_ratio = real_count / ai_count
        print(f"   Balance ratio (Real/AI): {balance_ratio:.2f}:1")
    
    return np.array(X), np.array(y)

def train_pitch_classifier():
    """Train and save a pitch-based classifier using ALL data"""
    print("=" * 60)
    print("🎯 TRAINING PITCH CLASSIFIER (ALL 4000+ SAMPLES)")
    print("=" * 60)
    
    # 1. Collect ALL data
    X, y = collect_all_training_data()
    
    if len(X) < 100:
        print("❌ Not enough data to train")
        return None
    
    # 2. Split data (keep 20% for testing)
    print("\n📈 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Count classes
    train_ai = np.sum(y_train == 0)
    train_human = np.sum(y_train == 1)
    test_ai = np.sum(y_test == 0)
    test_human = np.sum(y_test == 1)
    
    print(f"   Training - AI: {train_ai}, Human: {train_human}")
    print(f"   Testing - AI: {test_ai}, Human: {test_human}")
    
    # 3. Train Random Forest
    print("\n🌲 Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=200,           # More trees for larger dataset
        max_depth=15,               # Deeper trees for complex patterns
        min_samples_split=5,        # Prevent overfitting
        min_samples_leaf=2,
        class_weight='balanced',    # Handle any imbalance
        random_state=42,
        n_jobs=-1,                  # Use all CPU cores
        verbose=1                   # Show training progress
    )
    
    clf.fit(X_train, y_train)
    
    # 4. Evaluate
    print("\n📊 Evaluating classifier...")
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    
    print(f"   Training accuracy: {train_acc:.2%}")
    print(f"   Test accuracy: {test_acc:.2%}")
    
    # Detailed classification report
    y_pred = clf.predict(X_test)
    print(f"\n📋 Classification report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['AI', 'Human'],
                               digits=3))
    
    # 5. Feature importance
    print("\n🔍 Feature importance (what the model learned):")
    feature_names = ['Log Variance', 'Pitch Range', 'Mean Pitch', 
                    'Pitch Jumps', 'Pitch Smoothness', 'Voiced Ratio']
    
    importances = clf.feature_importances_
    for name, importance in sorted(zip(feature_names, importances), 
                                   key=lambda x: x[1], reverse=True):
        print(f"   {name:20} {importance:.3f}")
    
    # 6. Save models
    print("\n💾 Saving models...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save with timestamp
    model_filename = f'pitch_classifier_all_{timestamp}.pkl'
    joblib.dump(clf, model_filename)
    
    # Save as default for easy loading
    joblib.dump(clf, 'pitch_classifier.pkl')
    
    print(f"   {model_filename} (timestamped)")
    print(f"   pitch_classifier.pkl (default)")
    
    return clf

def load_pitch_classifier(model_path='pitch_classifier.pkl'):
    """Load trained pitch classifier"""
    if os.path.exists(model_path):
        try:
            clf = joblib.load(model_path)
            print(f"✅ Loaded pitch classifier from {model_path}")
            return clf
        except Exception as e:
            print(f"❌ Failed to load classifier: {e}")
    return None

def predict_with_pitch(audio_path, classifier=None):
    """
    Predict Human/AI using pitch features.
    Returns: (prediction, confidence, features)
    prediction: 0=AI, 1=Human
    confidence: probability of prediction (0-1)
    features: array of 6 pitch features
    """
    if classifier is None:
        classifier = load_pitch_classifier()
        if classifier is None:
            return None, None, None
    
    features = extract_pitch_features(audio_path)
    if features is None:
        return None, None, None
    
    # Get prediction and probabilities
    prediction = classifier.predict([features])[0]  # 0=AI, 1=Human
    probabilities = classifier.predict_proba([features])[0]
    
    confidence = probabilities[prediction]  # Confidence of predicted class
    
    return prediction, confidence, features

def analyze_audio_with_pitch(audio_path, classifier=None):
    """Complete analysis of audio file using pitch features"""
    if classifier is None:
        classifier = load_pitch_classifier()
    
    if classifier is None:
        print("❌ No classifier available")
        return
    
    print(f"\n🔍 Analyzing: {os.path.basename(audio_path)}")
    print("-" * 40)
    
    pred, conf, feats = predict_with_pitch(audio_path, classifier)
    
    if pred is not None:
        label = "HUMAN" if pred == 1 else "AI"
        print(f"Pitch-based prediction: {label}")
        print(f"Confidence: {conf:.1%}")
        
        # Show individual features
        feature_names = ['Log Variance', 'Pitch Range', 'Mean Pitch', 
                        'Pitch Jumps', 'Pitch Smoothness', 'Voiced Ratio']
        
        print("\nPitch features:")
        for name, value in zip(feature_names, feats):
            print(f"  {name:20} {value:.4f}")
        
        return pred, conf, feats
    else:
        print("❌ Could not extract pitch features")
        return None

if __name__ == "__main__":
    # Train a new classifier using ALL data
    clf = train_pitch_classifier()
    
    # Test with sample files
    print("\n" + "=" * 60)
    print("🧪 TESTING WITH SAMPLE FILES")
    print("=" * 60)
    
    test_files = []
    test_demo_dir = "test_demo"
    if os.path.exists(test_demo_dir):
        test_files = [os.path.join(test_demo_dir, f) for f in os.listdir(test_demo_dir) 
                     if f.endswith('.wav')][:3]
    
    if test_files and clf is not None:
        for test_file in test_files:
            analyze_audio_with_pitch(test_file, clf)
    else:
        print("No test files found or classifier not trained.")