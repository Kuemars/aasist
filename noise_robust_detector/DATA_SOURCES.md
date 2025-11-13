# Data Sources & Setup

## Real Voices (200 samples)
**Source**: LibriSpeech train-clean-100  
**Download**: Run `scripts/download_librispeech.py`  
**Size**: ~500MB (processed)  
**Location**: `data/raw/clean_real/`  
**Format**: 4.04s, 16kHz, 64600 samples per file

## AI Voices (Planned: 200 samples)
**Sources**: ElevenLabs, gTTS, Azure TTS, Amazon Polly  
**Script**: `scripts/plan_ai_voice_collection.py`  
**Target**: Match real voice count and quality

## Notes
- Data files are excluded from Git (see .gitignore)
- Download scripts provided for reproducibility
- All data processed to consistent format
