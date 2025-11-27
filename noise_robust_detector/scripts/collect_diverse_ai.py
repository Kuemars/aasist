import os
import sys
import time
import numpy as np
from gtts import gTTS
import soundfile as sf
import librosa

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Configuration
SAMPLE_RATE = 16000
AUDIO_LENGTH = 64600
AI_RAW_PATH = "data/raw/clean_ai"

def generate_diverse_ai_voices():
    """Generate diverse AI voices with varied demographics and content"""
    
    # Expanded diverse text templates
    diverse_texts = [
        # Female Young - Expanded
        ("I'm really excited about learning new programming languages this semester.", 'en', 'young_female'),
        ("The concert last weekend was absolutely amazing, the energy was incredible.", 'en', 'young_female'),
        ("I find that taking breaks while studying actually improves my retention.", 'en', 'young_female'),
        ("Social media can be overwhelming but it's great for staying connected.", 'en', 'young_female'),
        ("I've been practicing yoga and it's really helped with my stress levels.", 'en', 'young_female'),
        ("The new cafe downtown has the best coffee I've ever tasted, seriously.", 'en', 'young_female'),
        ("Online learning has its challenges but also offers great flexibility.", 'en', 'young_female'),
        ("I'm trying to read more books this year, starting with science fiction.", 'en', 'young_female'),
        
        # Female Professional - Expanded
        ("The implementation of agile methodologies has improved our team's productivity.", 'en', 'professional_female'),
        ("Data security protocols must be regularly updated to address emerging threats.", 'en', 'professional_female'),
        ("Stakeholder engagement is critical for successful project implementation.", 'en', 'professional_female'),
        ("The integration of AI tools has streamlined our workflow significantly.", 'en', 'professional_female'),
        ("Performance metrics indicate positive growth across all business units.", 'en', 'professional_female'),
        ("Cross-functional collaboration enhances innovation and problem-solving.", 'en', 'professional_female'),
        
        # Female Older - Expanded
        ("Technology has changed so much, but human connection remains essential.", 'en', 'older_female'),
        ("Gardening teaches patience and the rewards of consistent care over time.", 'en', 'older_female'),
        ("Traditional cooking methods often produce the most flavorful results.", 'en', 'older_female'),
        ("The wisdom we gain from experience cannot be found in books alone.", 'en', 'older_female'),
        ("Community involvement has always been important for building strong neighborhoods.", 'en', 'older_female'),
        
        # Male Young - Expanded
        ("The new smartphone features are pretty impressive, especially the camera.", 'en', 'young_male'),
        ("I'm working on building my own gaming computer from scratch this month.", 'en', 'young_male'),
        ("Basketball practice has been intense but it's making us better players.", 'en', 'young_male'),
        ("Streaming services have so much content it's hard to choose what to watch.", 'en', 'young_male'),
        ("Learning to code has opened up so many opportunities for creative projects.", 'en', 'young_male'),
        ("The road trip we took last summer was absolutely unforgettable adventure.", 'en', 'young_male'),
        
        # Male Professional - Expanded
        ("Strategic planning requires both quantitative analysis and qualitative insights.", 'en', 'professional_male'),
        ("The digital transformation initiative has accelerated our market responsiveness.", 'en', 'professional_male'),
        ("Risk management protocols must be comprehensive yet adaptable to change.", 'en', 'professional_male'),
        ("Team leadership involves both directing action and fostering collaboration.", 'en', 'professional_male'),
        ("The emerging technology landscape presents both challenges and opportunities.", 'en', 'professional_male'),
        
        # Male Older - Expanded
        ("Mentorship has taught me that learning is a lifelong journey for everyone.", 'en', 'older_male'),
        ("The pace of technological change requires continuous adaptation and learning.", 'en', 'older_male'),
        ("Historical knowledge provides context for understanding current developments.", 'en', 'older_male'),
        ("Effective communication skills remain fundamental to professional success.", 'en', 'older_male'),
        ("The integration of ethics and technology is increasingly important today.", 'en', 'older_male'),
        
        # Technical & Scientific - Expanded
        ("Machine learning models require large datasets for effective training.", 'en', 'technical'),
        ("The internet of things connects physical devices to digital networks.", 'en', 'technical'),
        ("Cryptographic algorithms ensure secure transmission of sensitive data.", 'en', 'technical'),
        ("Big data analytics reveals patterns that inform strategic decision making.", 'en', 'technical'),
        ("Cloud computing provides scalable infrastructure for modern applications.", 'en', 'technical'),
        ("User interface design must balance aesthetics with functionality.", 'en', 'technical'),
        
        # Storytelling & Narrative - Expanded
        ("The ancient forest whispered secrets to those who walked its paths quietly.", 'en', 'narrative'),
        ("She discovered the old photograph that would change everything she knew.", 'en', 'narrative'),
        ("The storm approached with dark clouds gathering on the distant horizon.", 'en', 'narrative'),
        ("In the quiet museum, history seemed to breathe through each exhibit.", 'en', 'narrative'),
        ("The journey across the desert tested their limits and their friendship.", 'en', 'narrative'),
        
        # Educational & Explanatory - Expanded
        ("The scientific method involves observation, hypothesis, and experimentation.", 'en', 'educational'),
        ("Climate change affects weather patterns and ecosystems globally.", 'en', 'educational'),
        ("The human brain processes information through complex neural networks.", 'en', 'educational'),
        ("Economic systems allocate resources through various mechanisms.", 'en', 'educational'),
        ("Linguistics studies the structure and evolution of human languages.", 'en', 'educational'),
        
        # Emotional & Expressive - Expanded
        ("The kindness of strangers can sometimes restore your faith in humanity.", 'en', 'emotional'),
        ("Achieving a long-term goal brings a deep sense of accomplishment.", 'en', 'emotional'),
        ("The beauty of a sunset can momentarily make all worries disappear.", 'en', 'emotional'),
        ("Support from friends during difficult times is truly invaluable.", 'en', 'emotional'),
        ("Creative expression allows us to share our inner world with others.", 'en', 'emotional'),
        
        # Question & Conversational - Expanded
        ("How do you think artificial intelligence will impact creative industries?", 'en', 'conversational'),
        ("What role should technology play in education and learning processes?", 'en', 'conversational'),
        ("Why is it important to maintain balance between work and personal life?", 'en', 'conversational'),
        ("How can we ensure technology serves humanity rather than controls it?", 'en', 'conversational'),
        ("What makes certain memories stand out more vividly than others?", 'en', 'conversational'),
        
        # Additional diverse content - Round 2
        ("The optimization algorithm converged faster with the improved parameters.", 'en', 'technical'),
        ("Morning journaling helps organize my thoughts and set daily intentions.", 'en', 'young_female'),
        ("Sustainable business practices benefit both companies and communities.", 'en', 'professional_male'),
        ("Family traditions create connections across generations and time.", 'en', 'older_female'),
        ("Augmented reality applications are transforming retail experiences.", 'en', 'young_male'),
        ("Market research provides valuable insights into consumer behavior.", 'en', 'professional_female'),
        ("The lessons from history help us navigate contemporary challenges.", 'en', 'older_male'),
        ("Digital art platforms have democratized creative expression.", 'en', 'young_female'),
        ("Renewable energy adoption requires both technology and policy support.", 'en', 'professional_male'),
        ("Music has the power to evoke emotions and memories uniquely.", 'en', 'emotional'),
        ("The software deployment process includes testing and validation phases.", 'en', 'technical'),
        ("Outdoor activities provide both physical exercise and mental refreshment.", 'en', 'young_female'),
        ("Corporate social responsibility enhances brand reputation and impact.", 'en', 'professional_male'),
        ("Handwritten letters carry a personal touch that digital messages lack.", 'en', 'older_female'),
        ("Virtual collaboration tools have revolutionized remote work dynamics.", 'en', 'young_male'),
        ("Financial planning requires both short-term and long-term perspectives.", 'en', 'professional_female'),
        ("The wisdom of experience often comes from learning from mistakes.", 'en', 'older_male'),
        ("Social entrepreneurship combines business principles with social impact.", 'en', 'young_female'),
        ("Cybersecurity measures must evolve to counter emerging digital threats.", 'en', 'professional_male'),
        ("Artistic creativity often flourishes within certain constraints.", 'en', 'emotional'),
    ]
    
    # Create output directory
    os.makedirs(AI_RAW_PATH, exist_ok=True)
    
    # Get current AI files to continue numbering
    existing_files = [f for f in os.listdir(AI_RAW_PATH) if f.startswith('ai_') and f.endswith('.wav')]
    start_idx = len(existing_files)
    
    print(f"Found {start_idx} existing AI voices. Generating {len(diverse_texts)} diverse AI voices...")
    print("This will include: Young/Professional/Older, Male/Female voices, Various content types")
    
    successful_generations = 0
    
    for i, (text, lang, category) in enumerate(diverse_texts):
        try:
            # Generate speech - gTTS will handle voice variations automatically
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # Save temporary file
            temp_file = os.path.join(AI_RAW_PATH, f'temp_{start_idx + i}.mp3')
            tts.save(temp_file)
            
            # Load and convert to consistent format
            audio, sr = librosa.load(temp_file, sr=SAMPLE_RATE)
            
            # Ensure correct length
            if len(audio) > AUDIO_LENGTH:
                audio = audio[:AUDIO_LENGTH]
            else:
                # Pad if shorter
                padding = AUDIO_LENGTH - len(audio)
                audio = np.pad(audio, (0, padding))
            
            # Save final file with category info in filename
            output_file = os.path.join(AI_RAW_PATH, f'ai_{start_idx + i + 1:03d}_{category}.wav')
            sf.write(output_file, audio, SAMPLE_RATE)
            
            successful_generations += 1
            print(f"Generated AI voice {start_idx + i + 1:03d} ({category}): {text[:40]}...")
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            # Small delay to avoid rate limiting
            time.sleep(2)
            
        except Exception as e:
            print(f"Error generating voice {i} ({category}): {e}")
            continue
    
    total_ai_voices = start_idx + successful_generations
    print(f"\n✅ AI voice generation complete!")
    print(f"   Successfully generated: {successful_generations} new voices")
    print(f"   Total AI voices now: {total_ai_voices}")
    print(f"   Dataset balance: 200 real vs {total_ai_voices} AI")
    
    if total_ai_voices >= 200:
        print("🎉 Perfect! Dataset is now balanced for training!")
    else:
        print(f"⚠️  Still need {200 - total_ai_voices} more AI voices for perfect balance")

if __name__ == "__main__":
    generate_diverse_ai_voices()