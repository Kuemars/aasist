# scripts/comprehensive_ai_collection.py
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import project_config_extended as config

class ComprehensiveAICollection:
    def __init__(self):
        self.target_count = 943  # Match real voices count
        
    def get_current_ai_count(self):
        """Count current AI voices in clean_ai directory"""
        ai_dir = config.AI_RAW_DIR
        if ai_dir.exists():
            ai_files = list(ai_dir.glob("*.wav"))
            return len(ai_files)
        return 0
    
    def get_current_real_count(self):
        """Count current real voices"""
        real_dir = config.REAL_RAW_DIR
        if real_dir.exists():
            real_files = list(real_dir.glob("*.wav"))
            return len(real_files)
        return 0
    
    def collect_from_eleven_labs(self):
        """Collect from Eleven Labs dataset"""
        print("\n📥 1. COLLECTING FROM ELEVEN LABS RESPEECH")
        try:
            from scripts.download_eleven_labs import main as download_eleven
            files = download_eleven()
            return files if files else []
        except Exception as e:
            print(f"❌ Eleven Labs failed: {e}")
            return []
    
    def collect_from_asvspoof(self):
        """Collect from ASVspoof dataset"""
        print("\n📥 2. COLLECTING FROM ASVSPOOF 2019")
        try:
            from scripts.download_asvspoof import main as download_asvspoof
            files = download_asvspoof()
            return files if files else []
        except Exception as e:
            print(f"❌ ASVspoof failed: {e}")
            return []
    
    def collect_existing_gtts(self):
        """Collect existing gTTS voices"""
        print("\n📥 3. COLLECTING EXISTING GTTS VOICES")
        ai_dir = config.AI_RAW_DIR
        if ai_dir.exists():
            gtts_files = list(ai_dir.glob("ai_*.wav"))
            print(f"✅ Found {len(gtts_files)} existing gTTS voices")
            return gtts_files
        return []
    
    def generate_additional_gtts(self, needed_count):
        """Generate additional gTTS voices if needed"""
        if needed_count <= 0:
            return []
            
        print(f"\n📥 4. GENERATING {needed_count} ADDITIONAL GTTS VOICES")
        try:
            from scripts.collect_gtts_ai import generate_ai_voices
            current_count = self.get_current_ai_count()
            new_voices = generate_ai_voices(needed_count, start_index=current_count + 1)
            print(f"✅ Generated {len(new_voices)} additional gTTS voices")
            return new_voices
        except Exception as e:
            print(f"❌ Additional gTTS generation failed: {e}")
            return []
    
    def consolidate_all_ai_voices(self, all_files):
        """Consolidate all AI voices to clean_ai with consistent naming"""
        print("\n🔄 CONSOLIDATING ALL AI VOICES")
        
        ai_dir = config.AI_RAW_DIR
        ai_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove duplicates and get unique files
        unique_files = list(set(all_files))
        unique_files.sort()
        
        print(f"📊 Total unique AI files: {len(unique_files)}")
        
        # Copy all files with consistent naming
        consolidated_files = []
        for i, source_file in enumerate(tqdm(unique_files, desc="Consolidating")):
            try:
                target_filename = f"ai_{i+1:04d}.wav"
                target_path = ai_dir / target_filename
                
                # If file is already in target directory with correct name, keep it
                if source_file.parent == ai_dir and source_file.name == target_filename:
                    consolidated_files.append(target_path)
                    continue
                
                # Otherwise copy the file
                shutil.copy2(source_file, target_path)
                consolidated_files.append(target_path)
                
            except Exception as e:
                print(f"❌ Error consolidating {source_file}: {e}")
        
        return consolidated_files
    
    def main(self):
        """Main collection function"""
        print("🚀 COMPREHENSIVE AI VOICE COLLECTION")
        print("=" * 50)
        
        # Get current counts
        current_real = self.get_current_real_count()
        current_ai = self.get_current_ai_count()
        
        print(f"📊 CURRENT DATASET STATUS:")
        print(f"   Real voices: {current_real}")
        print(f"   AI voices: {current_ai}")
        print(f"   Needed AI voices: {self.target_count - current_ai}")
        
        if current_ai >= self.target_count:
            print("🎉 AI dataset already balanced!")
            return True
        
        # Collect from all sources
        all_ai_files = []
        
        # Add existing gTTS voices
        existing_gtts = self.collect_existing_gtts()
        all_ai_files.extend(existing_gtts)
        
        # Add Eleven Labs
        eleven_files = self.collect_from_eleven_labs()
        all_ai_files.extend(eleven_files)
        
        # Add ASVspoof
        asvspoof_files = self.collect_from_asvspoof()
        all_ai_files.extend(asvspoof_files)
        
        # Check if we need more
        current_total = len(all_ai_files)
        needed = max(0, self.target_count - current_total)
        
        if needed > 0:
            additional_gtts = self.generate_additional_gtts(needed)
            all_ai_files.extend(additional_gtts)
        
        # Consolidate all files
        final_files = self.consolidate_all_ai_voices(all_ai_files)
        
        # Final status
        final_ai_count = len(final_files)
        final_real_count = self.get_current_real_count()
        
        print("\n🎯 FINAL DATASET STATUS:")
        print(f"   Real voices: {final_real_count}")
        print(f"   AI voices: {final_ai_count}")
        print(f"   Balance: {final_real_count}:{final_ai_count}")
        
        if final_ai_count >= self.target_count:
            print("🎉 SUCCESS! AI dataset is balanced and ready for training!")
            return True
        else:
            print(f"⚠️  Partial success: {final_ai_count}/{self.target_count} AI voices")
            return False

def main():
    collector = ComprehensiveAICollection()
    success = collector.main()
    return success

if __name__ == "__main__":
    main()