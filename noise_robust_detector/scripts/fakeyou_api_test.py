import requests
import time
import json

def debug_fakeyou_api():
    """Debug script to find working voices and fix the issues"""
    
    BASE_URL = "https://api.fakeyou.com"
    
    print("🔍 DEBUG: Testing FakeYou API")
    print("=" * 50)
    
    # 1. Test API connectivity
    print("1. Testing API connectivity...")
    response = requests.get(f"{BASE_URL}/tts/list")
    print(f"   Status: {response.status_code}")
    
    if response.status_code != 200:
        print("❌ API is down")
        return
    
    # 2. Get voices and check specific ones
    data = response.json()
    voices = data.get("models", [])
    print(f"2. Found {len(voices)} total voices")
    
    # 3. Look for verified working voices
    print("3. Searching for verified working voices...")
    
    # Common working voice patterns
    working_patterns = [
        "en", "english", "us", "uk", "american", "british",
        "female", "male", "neutral", "default"
    ]
    
    working_voices = []
    for voice in voices[:100]:  # Check first 100
        title_lower = voice["title"].lower()
        if any(pattern in title_lower for pattern in working_patterns):
            working_voices.append(voice)
            if len(working_voices) >= 10:
                break
    
    print(f"   Found {len(working_voices)} potential working voices")
    
    # 4. Test one voice with simple text
    if working_voices:
        test_voice = working_voices[0]
        print(f"4. Testing voice: {test_voice['title']}")
        print(f"   Token: {test_voice['model_token']}")
        
        # Test with very simple text
        payload = {
            "tts_model_token": test_voice["model_token"],
            "uuid_idempotency_token": f"debug_test_{int(time.time())}",
            "inference_text": "Hello world."  # Very simple text
        }
        
        print("   Submitting test job...")
        response = requests.post(f"{BASE_URL}/tts/inference", json=payload)
        
        if response.status_code == 200:
            job_data = response.json()
            print(f"   ✅ Job submitted: {job_data['inference_job_token']}")
            print("   This means the voice is valid!")
            
            # Show working voices
            print("\n🎯 POTENTIAL WORKING VOICES:")
            for i, voice in enumerate(working_voices):
                print(f"   {i+1}. {voice['title']}")
                
        else:
            print(f"   ❌ Job submission failed: {response.status_code}")
            print(f"   Response: {response.text}")
    
    # 5. Check if there are any known good voices
    print("\n5. Checking for known good voices...")
    known_good_tokens = [
        "tm_bb8d58c375",  # Common test voice
        "tm_2c8d58c375",  # Another common one
    ]
    
    for token in known_good_tokens:
        print(f"   Testing known token: {token}")
        payload = {
            "tts_model_token": token,
            "uuid_idempotency_token": f"known_test_{int(time.time())}",
            "inference_text": "Test."
        }
        
        response = requests.post(f"{BASE_URL}/tts/inference", json=payload)
        if response.status_code == 200:
            print(f"   ✅ Known voice works!")
            break
        else:
            print(f"   ❌ Known voice failed: {response.status_code}")

if __name__ == "__main__":
    debug_fakeyou_api()