import asyncio
import random
import soundfile as sf
import numpy as np
from fakeyou import FakeYou

async def debug_fakeyou_methods():
    """Debug to find the correct methods"""
    print("🔍 Debugging FakeYou library methods...")
    
    fakeyou = FakeYou()
    
    # Check what methods are available
    print("Available methods:")
    for method in dir(fakeyou):
        if not method.startswith('_'):
            print(f"  - {method}")
    
    # Try to discover voices
    try:
        print("\n🎯 Trying to discover voices...")
        # Common method names in TTS libraries
        if hasattr(fakeyou, 'list_voices'):
            voices = await fakeyou.list_voices()
            print(f"list_voices: {len(voices) if voices else 'None'}")
        elif hasattr(fakeyou, 'get_voices'):
            voices = await fakeyou.get_voices()
            print(f"get_voices: {len(voices) if voices else 'None'}")
        elif hasattr(fakeyou, 'voices'):
            voices = fakeyou.voices
            print(f"voices: {len(voices) if voices else 'None'}")
        else:
            print("❌ No voice discovery methods found")
            
    except Exception as e:
        print(f"❌ Voice discovery failed: {e}")

# Run the debug
asyncio.run(debug_fakeyou_methods())