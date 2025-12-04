import requests

# Test which models are available
test_models = [
    "microsoft/speecht5_tts",
    "facebook/mms-tts",
    "espnet/kan-bayashi_ljspeech_vits",
    "speechbrain/tts-tacotron2-ljspeech"
]

API_URL = "https://router.huggingface.co/hf-inference"

for model in test_models:
    try:
        response = requests.post(API_URL, json={
            "inputs": "Test",
            "model": model
        })
        print(f"{model}: {response.status_code}")
        if response.status_code != 200:
            print(f"   Error: {response.text[:100]}")
    except Exception as e:
        print(f"{model}: Failed - {e}")