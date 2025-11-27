import os
import sys
import torch
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import subprocess
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from models.AASIST import Model
    import yaml

    print("=" * 60)
    print("🎯 AI VOICE DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initializing AASIST Detector on {device}")
    
    # FIXED PATH: Use absolute path to config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config/AASIST.conf')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model = Model(config['model_config']).to(device)
    
    # FIXED PATH: Use correct path to fine-tuned model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models/aasist_fine_tuned.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Loaded fine-tuned model weights")
    print("✅ AASIST Detector ready!")

    def get_ffmpeg_path():
        """Get path to ffmpeg executable - bundled with project"""
        # Look for ffmpeg in project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        ffmpeg_path = os.path.join(project_root, 'ffmpeg.exe')
        
        if os.path.exists(ffmpeg_path):
            return ffmpeg_path
        
        # Also check in scripts directory
        scripts_ffmpeg = os.path.join(os.path.dirname(__file__), 'ffmpeg.exe')
        if os.path.exists(scripts_ffmpeg):
            return scripts_ffmpeg
        
        # Try system ffmpeg as fallback
        return 'ffmpeg'

    def convert_audio_to_wav(audio_path):
        """Convert any audio format to WAV using bundled ffmpeg"""
        try:
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_wav = temp_file.name
            
            # Get ffmpeg path
            ffmpeg_path = get_ffmpeg_path()
            
            # Convert using ffmpeg
            cmd = [
                ffmpeg_path, 
                '-i', audio_path, 
                '-ac', '1',           # Mono
                '-ar', '16000',       # 16kHz
                '-acodec', 'pcm_s16le', # 16-bit WAV
                '-y',                 # Overwrite output
                temp_wav
            ]
            
            # Run conversion (suppress output)
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   ✅ Converted to WAV successfully")
                return temp_wav
            else:
                print(f"   ⚠️  Conversion failed: {result.stderr[:100]}...")
                return None
                
        except Exception as e:
            print(f"   ⚠️  Conversion error: {e}")
            return None

    def load_any_audio(audio_path, target_sr=16000, target_length=64600):
        """Load any audio format and convert to model-compatible format"""
        try:
            # For problematic formats (M4A, etc.), convert to WAV first
            problematic_extensions = {'.m4a', '.aac', '.m4b', '.mp4', '.wma'}
            file_ext = os.path.splitext(audio_path.lower())[1]
            
            if file_ext in problematic_extensions:
                print(f"   🔄 Converting {file_ext.upper()} to WAV...")
                wav_path = convert_audio_to_wav(audio_path)
                if wav_path:
                    # Load the converted WAV file
                    audio, sr = librosa.load(wav_path, sr=target_sr)
                    # Clean up temporary file
                    try:
                        os.unlink(wav_path)
                    except:
                        pass
                else:
                    # Fallback: try direct loading
                    print(f"   ⚠️  Trying direct load as fallback...")
                    audio, sr = librosa.load(audio_path, sr=target_sr)
            else:
                # Load standard formats directly
                audio, sr = librosa.load(audio_path, sr=target_sr)
            
            print(f"   📥 Loaded: {len(audio)} samples")
            
            # Ensure correct length
            if len(audio) > target_length:
                audio = audio[:target_length]
                print(f"   ⚠️  Trimmed to {target_length} samples")
            else:
                padding = target_length - len(audio)
                audio = np.pad(audio, (0, padding))
                print(f"   ⚠️  Padded to {target_length} samples")
            
            return torch.FloatTensor(audio).unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            print(f"❌ Error loading {audio_path}: {e}")
            return None

    def predict_audio(audio_path):
        """Predict if audio is real or AI"""
        print(f"   🔍 Processing: {os.path.basename(audio_path)}")
        audio_tensor = load_any_audio(audio_path)
        if audio_tensor is None:
            return "ERROR", 0.0, {}

        with torch.no_grad():
            output_tuple = model(audio_tensor)
            scores = output_tuple[1]  # This is the [1, 2] tensor

            # Apply softmax to get probabilities
            probabilities = torch.softmax(scores, dim=1)
            human_prob = probabilities[0][0].item() * 100
            ai_prob = probabilities[0][1].item() * 100

            # Calculate confidence metrics
            confidence_metrics = {
                'human_prob': human_prob,
                'ai_prob': ai_prob,
                'confidence_gap': abs(human_prob - ai_prob),
            }

            if ai_prob > human_prob:
                return "AI_GENERATED", ai_prob, confidence_metrics
            else:
                return "HUMAN", human_prob, confidence_metrics

    # Create test directory
    test_dir = "test_demo"
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"\n📁 Created test directory: {test_dir}")  # FIXED: was test_demo, now test_dir
    print("💡 Supported audio formats: WAV, MP3, M4A, AAC, FLAC, OGG, etc.")
    print("   Just drop any audio files in the test_demo directory!")

    # Find ALL audio files in test directory and current directory
    audio_extensions = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg', '.wma', '.m4b', '.mp4'}
    audio_files = []
    
    # Check test directory
    for file in os.listdir(test_dir):
        file_path = os.path.join(test_dir, file)
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(file_path)
    
    # Also check current directory for quick testing
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(file)

    if not audio_files:
        print(f"⚠️  No audio files found!")
        print(f"💡 Please add audio files to:")
        print(f"   - {test_dir}/ (for organized testing)")
        print(f"   - Current directory (for quick testing)")
        print(f"📋 Supported formats: {', '.join(audio_extensions)}")
        sys.exit(1)

    print(f"\n🔍 Found {len(audio_files)} audio files:")
    for file in audio_files:
        print(f"   - {os.path.basename(file)}")

    print("\n" + "=" * 60)
    print("📊 RUNNING DETECTION...")
    print("=" * 60)

    results = []

    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        print(f"\n🎵 Analyzing: {filename}")
        
        result, confidence, metrics = predict_audio(audio_file)
        
        if "ERROR" in result:
            print(f"   ❌ Failed to analyze")
            continue
            
        if "AI_GENERATED" in result:
            print(f"   Result: 🔴 {result}")
            print(f"   Confidence: {confidence:.1f}%")
        else:
            print(f"   Result: 🟢 {result}")  
            print(f"   Confidence: {confidence:.1f}%")
            
        print(f"   Human Probability: {metrics['human_prob']:.1f}%")
        print(f"   AI Probability: {metrics['ai_prob']:.1f}%")
        
        results.append((filename, result, confidence))

    print("\n" + "=" * 60)
    print("📈 SUMMARY")
    print("=" * 60)

    for filename, result, confidence in results:
        if "AI_GENERATED" in result:
            print(f"🔴 AI: {filename} ({confidence:.1f}% confidence)")
        else:
            print(f"🟢 HUMAN: {filename} ({confidence:.1f}% confidence)")

    print(f"\n✅ Demonstration complete! Model successfully analyzed {len(results)} files.")

except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
