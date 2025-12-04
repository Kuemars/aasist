import subprocess
import os
import random
import csv
from pathlib import Path
from datetime import datetime

# ========== CONFIGURATION ==========
# UPDATE THESE PATHS TO MATCH YOUR SYSTEM
PIPER_EXE_PATH = r"D:\sk13382\piper_windows_amd64\piper\piper.exe"
VOICES_FOLDER_PATH = r"D:\sk13382\piper_windows_amd64\piper\voices"
SAMPLES_DIR_PATH = r"D:\sk13382\piper_windows_amd64\piper\Samples"  # Your specified folder
TOTAL_VOICES_NEEDED = 600
# ===================================

# 1. TEXT CORPUS FOR ADVANCED DETECTION TRAINING
text_categories = {
    "EMOTIONAL": [
        "I absolutely cannot believe this is happening right now!",
        "This is utterly unacceptable and must be addressed immediately.",
        "Oh my goodness, that's the most wonderful news I've heard all year!",
        "Please... just leave me alone. I can't deal with this anymore.",
        "I'm so excited about the upcoming conference and all the possibilities!",
        "This is absolutely devastating news. I don't know how to process it.",
        "How dare you speak to me in that tone of voice? Show some respect!",
        "I've never been more proud of a team in my entire career. Outstanding work!",
        "The anticipation is killing me! I can barely contain my excitement.",
        "This level of negligence is borderline criminal. Someone must be held accountable."
    ],
    
    "TECHNICAL": [
        "The convolutional neural network's gradient descent converged after 300 epochs.",
        "Pseudorandom cryptographic seeds require entropy from multiple hardware sources.",
        "Spectral centroid analysis reveals artifacts in the mel-frequency cepstral coefficients.",
        "Quantum decoherence presents a significant challenge for error correction in qubits.",
        "The transformer architecture utilizes multi-head attention mechanisms for context.",
        "Adversarial training with projected gradient descent improves model robustness.",
        "Homomorphic encryption allows computation on encrypted data without decryption.",
        "The Bellman equation provides the foundation for dynamic programming in RL.",
        "Non-negative matrix factorization decomposes spectrograms into basis components.",
        "Differential privacy introduces statistical noise to preserve individual anonymity."
    ],
    
    "CONVERSATIONAL": [
        "So, um, like, I was thinking we could, you know, maybe try a different approach?",
        "Okay, well, actually, I mean, if you look at it from that perspective...",
        "Right, so basically, what I'm saying is, we need to, uh, reconsider the timeline.",
        "Hey, how's it going? Did you get a chance to look at those documents I sent over?",
        "Wait, sorry, can you repeat that? I didn't quite catch the last part.",
        "Anyway, long story short, we ended up missing the deadline by just a few hours.",
        "You know what? Now that I think about it, that might not be such a bad idea.",
        "To be completely honest with you, I'm not entirely sure about that specific detail.",
        "Hmm, let me see... I suppose we could try it the other way around, maybe?",
        "Alright, look, here's the thing: we're going to need more time on this project."
    ],
    
    "MULTILINGUAL": [
        "The rendezvous is set for mañana at the café near the park. Compris?",
        "We need to consider the Zeitgeist of this entire debacle, mais oui?",
        "C'est la vie, but honestly, the schadenfreude is just too much to bear.",
        "The doppelgänger had a certain je ne sais quoi that was quite unsettling.",
        "Let's have a tête-à-tête to discuss the modus operandi for this operation.",
        "It's a real faux pas to discuss finances in such a cavalier manner, honestly.",
        "The wanderlust is strong, but my carte blanche has some serious restrictions.",
        "There's a certain dolce far niente about Sundays that I truly appreciate.",
        "The smorgasbord of options created a sense of deja vu that was uncanny.",
        "His alibi had a certain coup de grâce that made it completely unbelievable."
    ],
    
    "ENUMERATION": [
        "One, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve.",
        "Alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa.",
        "First, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth.",
        "January, February, March, April, May, June, July, August, September, October.",
        "Red, orange, yellow, green, blue, indigo, violet, purple, pink, magenta.",
        "North, south, east, west, northeast, northwest, southeast, southwest.",
        "Circle, square, triangle, rectangle, pentagon, hexagon, octagon, decagon.",
        "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto.",
        "Piano, forte, crescendo, decrescendo, allegro, adagio, presto, lento.",
        "Binary, ternary, quaternary, quinary, senary, septenary, octonary, nonary."
    ],
    
    "QUESTION_ANSWER": [
        "What is the capital of France? The capital of France is Paris.",
        "How does a neural network learn? It learns through backpropagation and gradient descent.",
        "Why is the sky blue? The sky appears blue due to Rayleigh scattering of sunlight.",
        "What is photosynthesis? Photosynthesis is how plants convert light energy to chemical energy.",
        "Who wrote Hamlet? William Shakespeare wrote the tragedy Hamlet.",
        "How do vaccines work? Vaccines stimulate the immune system to recognize and fight pathogens.",
        "What causes earthquakes? Earthquakes are caused by sudden slip along geological faults.",
        "What is blockchain? Blockchain is a distributed ledger technology for secure transactions.",
        "How do airplanes fly? Airplanes fly through lift generated by wing shape and airspeed.",
        "What is quantum entanglement? Quantum entanglement is when particles remain connected across distance."
    ],
    
    "PHONEME_CHALLENGE": [
        "The sixth sick sheik's sixth sheep's sick.",
        "Pad kid poured curd pulled cod.",
        "Which witch wished which wicked wish?",
        "Fuzzy Wuzzy was a bear. Fuzzy Wuzzy had no hair.",
        "She sells seashells by the seashore.",
        "Peter Piper picked a peck of pickled peppers.",
        "How can a clam cram in a clean cream can?",
        "I slit the sheet, the sheet I slit, and on the slitted sheet I sit.",
        "Lesser leather never weathered wetter weather better.",
        "A proper cup of coffee in a proper copper coffee pot."
    ]
}

# 2. DISCOVER ALL AVAILABLE VOICE MODELS
def discover_voice_models(voices_folder):
    """Find all valid voice model pairs (.onnx + .json)"""
    voice_models = []
    folder_path = Path(voices_folder)
    
    for onnx_file in folder_path.glob("*.onnx"):
        json_file = onnx_file.with_suffix('.onnx.json')
        if json_file.exists():
            voice_models.append({
                "name": onnx_file.stem.replace('.onnx', ''),
                "onnx_path": str(onnx_file),
                "json_path": str(json_file)
            })
            print(f"  ✓ Found: {onnx_file.name}")
        else:
            print(f"  ⚠️  Missing JSON for: {onnx_file.name}")
    
    return voice_models

# 3. PREPARE OUTPUT DIRECTORY WITH SMART NUMBERING
def prepare_output_directory(samples_dir):
    """Prepares the fixed Samples directory and finds the next file number."""
    samples_path = Path(samples_dir)
    
    # Create the directory if it doesn't exist
    samples_path.mkdir(exist_ok=True)
    print(f"📁 Output directory: {samples_path}")
    
    # Find the highest existing ai_xxxx.wav file number
    existing_files = list(samples_path.glob("ai_*.wav"))
    start_number = 0
    if existing_files:
        # Extract numbers from filenames and find the maximum
        numbers = []
        for f in existing_files:
            try:
                # Get the number from 'ai_0001.wav'
                num = int(f.stem.split('_')[1])
                numbers.append(num)
            except (IndexError, ValueError):
                continue  # Skip files that don't match the pattern
        if numbers:
            start_number = max(numbers) + 1
    
    print(f"🔢 Starting file count from: ai_{start_number:04d}.wav")
    return samples_path, start_number

# 4. GENERATION FUNCTION
def generate_with_piper(piper_exe, model_path, text, output_path):
    """Call piper.exe to generate speech"""
    command = [
        piper_exe,
        "--model", model_path,
        "--output_file", output_path
    ]
    
    try:
        result = subprocess.run(
            command,
            input=text.encode('utf-8'),
            capture_output=True,
            check=True,
            shell=True
        )
        return True, None
    except subprocess.CalledProcessError as e:
        error = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
        return False, error

# 5. MAIN GENERATION LOGIC
def generate_comprehensive_dataset():
    print("=" * 70)
    print("ADVANCED AI VOICE DATASET GENERATION")
    print("=" * 70)
    
    # Verify Piper exists
    if not os.path.exists(PIPER_EXE_PATH):
        print(f"❌ ERROR: piper.exe not found at:\n{PIPER_EXE_PATH}")
        return
    
    # Discover voice models
    print("\n🔍 Discovering voice models...")
    voice_models = discover_voice_models(VOICES_FOLDER_PATH)
    
    if not voice_models:
        print("❌ No valid voice models found. Download voices first.")
        return
    
    print(f"\n✅ Found {len(voice_models)} distinct voice models")
    
    # Prepare the fixed Samples directory with smart numbering
    output_dir, start_file_number = prepare_output_directory(SAMPLES_DIR_PATH)
    
    # Prepare CSV log inside the Samples directory
    csv_path = output_dir / "generation_log.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_id', 'voice_model', 'text_category', 'text', 'output_file', 'success']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    print(f"\n📊 Target: {TOTAL_VOICES_NEEDED} AI voice samples")
    print(f"🎯 Using {len(voice_models)} voice models × {len(text_categories)} text categories")
    print("\n" + "=" * 70)
    
    # Calculate distribution
    voices_per_model = TOTAL_VOICES_NEEDED // len(voice_models)
    remainder = TOTAL_VOICES_NEEDED % len(voice_models)
    
    print(f"Distribution: ~{voices_per_model} samples per voice model")
    
    # Flatten text corpus
    all_texts = []
    for category, texts in text_categories.items():
        for text in texts:
            all_texts.append((category, text))
    
    # Generation loop
    files_generated = 0
    files_failed = 0
    
    for voice_idx, voice in enumerate(voice_models):
        samples_for_this_voice = voices_per_model + (1 if voice_idx < remainder else 0)
        
        print(f"\n🎤 Voice {voice_idx+1}/{len(voice_models)}: {voice['name']}")
        print(f"   Generating {samples_for_this_voice} samples...")
        
        for sample_idx in range(samples_for_this_voice):
            if files_generated >= TOTAL_VOICES_NEEDED:
                break
            
            # Select text: cycle through categories for diversity
            text_category, text = all_texts[(files_generated) % len(all_texts)]
            
            # Create output filename: continues from existing count
            output_filename = f"ai_{(start_file_number + files_generated):04d}.wav"
            output_path = output_dir / output_filename
            
            # Show progress
            if files_generated % 50 == 0:
                print(f"   Progress: {files_generated}/{TOTAL_VOICES_NEEDED}")
            
            # Generate
            success, error = generate_with_piper(
                PIPER_EXE_PATH,
                voice['onnx_path'],
                text,
                str(output_path)
            )
            
            # Log result
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'file_id': start_file_number + files_generated,
                    'voice_model': voice['name'],
                    'text_category': text_category,
                    'text': text,
                    'output_file': output_filename,
                    'success': success
                })
            
            if success:
                files_generated += 1
            else:
                files_failed += 1
                print(f"   ❌ Failed: {error[:100]}")
    
    # Final report
    print("\n" + "=" * 70)
    print("✅ GENERATION COMPLETE")
    print("=" * 70)
    print(f"📊 Total generated: {files_generated}/{TOTAL_VOICES_NEEDED}")
    print(f"❌ Failures: {files_failed}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📋 Log file: {csv_path}")
    
    # Statistics
    if files_generated > 0:
        print(f"\n📈 Dataset Statistics:")
        print(f"   • Unique voice models: {len(voice_models)}")
        print(f"   • Text categories: {len(text_categories)}")
        print(f"   • Average samples per voice: {files_generated/len(voice_models):.1f}")
        print(f"   • Total text diversity: {len(all_texts)} unique sentences")
        print(f"   • File range: ai_{start_file_number:04d}.wav to ai_{(start_file_number + files_generated - 1):04d}.wav")
    
    print("\n🎯 This dataset is optimized for voice clone detection training.")
    print("   The diversity in voices and challenging text should help your model")
    print("   learn to identify subtle synthesis artifacts across various conditions.")
    print("=" * 70)

if __name__ == "__main__":
    generate_comprehensive_dataset()