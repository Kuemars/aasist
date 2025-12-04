import subprocess
import os

# === UPDATE THESE PATHS TO MATCH YOUR SYSTEM ===
# Path to the piper.exe file
PIPER_EXE_PATH = r"D:\sk13382\piper_windows_amd64\piper\piper.exe"
# Path to the folder containing your .onnx and .json voice files
VOICES_FOLDER_PATH = r"D:\sk13382\piper_windows_amd64\piper\voices"
# ================================================

# Voice models available in your voices folder
# (Add more here if you download additional voices)
voices = [
    {"model": "en_US-libritts-high.onnx", "name": "libritts_female"},
    {"model": "en_US-bryce-medium.onnx", "name": "bryce_male"},
]

# Diverse text samples for testing quality
text_samples = [
    	"The sixth sick sheik's sixth sheep's sick.",
    	"Pad kid poured curd pulled cod.",
        "Which witch wished which wicked wish?",
        "Fuzzy Wuzzy was a bear. Fuzzy Wuzzy had no hair.",
        "She sells seashells by the seashore."
]

def generate_speech(model_file, text, output_file):
    """Generates a WAV file using the Piper executable."""
    # Construct the full path to the model file
    model_path = os.path.join(VOICES_FOLDER_PATH, model_file)
    
    # Prepare the command for piper.exe
    command = [
        PIPER_EXE_PATH,
        "--model", model_path,
        "--output_file", output_file
    ]
    
    try:
        # Run piper.exe and pipe the text to it
        result = subprocess.run(
            command,
            input=text.encode(),
            capture_output=True,
            check=True,  # Raises an error if piper fails
            shell=True   # Often needed on Windows
        )
        print(f"  ✓ Generated: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error generating {output_file}:")
        if e.stderr:
            print(f"    Piper Error: {e.stderr.decode().strip()}")
        return False
    except FileNotFoundError:
        print(f"  ✗ Error: Could not find piper.exe at {PIPER_EXE_PATH}")
        print("    Please check the PIPER_EXE_PATH in the script.")
        return False

def main():
    """Main function to generate 20 test samples."""
    print("=" * 50)
    print("Piper TTS Quality Test - Generating 20 Samples")
    print("=" * 50)
    
    # Verify paths exist
    if not os.path.exists(PIPER_EXE_PATH):
        print(f"ERROR: piper.exe not found at:\n{PIPER_EXE_PATH}")
        return
    
    if not os.path.exists(VOICES_FOLDER_PATH):
        print(f"ERROR: Voices folder not found at:\n{VOICES_FOLDER_PATH}")
        return
    
    print(f"Using Piper from: {PIPER_EXE_PATH}")
    print(f"Using voices from: {VOICES_FOLDER_PATH}")
    print("-" * 50)
    
    samples_generated = 0
    
    for i in range(20):  # Generate exactly 20 samples
        # Cycle through voices and text samples
        voice = voices[i % len(voices)]
        text = text_samples[i % len(text_samples)]
        
        # Create the output filename as piper_ai_xxx.wav
        output_filename = f"piper_ai_{i:03d}.wav"
        
        print(f"\nSample {i+1}/20")
        print(f"  Voice: {voice['name']} ({voice['model']})")
        print(f"  Text: \"{text}\"")
        
        # Generate the speech file
        if generate_speech(voice["model"], text, output_filename):
            samples_generated += 1
    
    # Final summary
    print("\n" + "=" * 50)
    print(f"GENERATION COMPLETE")
    print(f"Successfully generated: {samples_generated}/20 samples")
    print(f"Files are saved in: {os.getcwd()}")
    
    if samples_generated < 20:
        print(f"\nNote: {20 - samples_generated} samples failed. Check errors above.")
    print("=" * 50)

if __name__ == "__main__":
    # Ensure the script runs from the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()