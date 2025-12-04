# scripts/launch_training.py
import os
import sys
import subprocess
import argparse

def check_gpu():
    """Check GPU availability and status"""
    import torch
    
    print("🔍 Checking GPU status...")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA is available")
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Test GPU speed
        print("\n⚡ Testing GPU performance...")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        x = torch.randn(10000, 10000).cuda()
        y = torch.randn(10000, 10000).cuda()
        
        start.record()
        z = torch.matmul(x, y)
        end.record()
        
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        
        print(f"   Matrix multiply: {elapsed:.2f} ms")
        print(f"   GFLOPs: {(2 * 10000**3) / (elapsed * 1e9):.2f}")
        
        return True
    else:
        print("❌ No GPU detected!")
        return False

def check_dataset():
    """Check dataset status"""
    print("\n🔍 Checking dataset...")
    
    real_path = r"D:\sk13382\DataTraining\aasist\noise_robust_detector\data\raw\clean_real"
    ai_path = r"D:\sk13382\DataTraining\aasist\noise_robust_detector\data\raw\clean_ai"
    
    if os.path.exists(real_path):
        real_files = [f for f in os.listdir(real_path) if f.startswith('real_') and f.endswith('.wav')]
        print(f"✅ Real voices: {len(real_files)} files")
    else:
        print(f"❌ Real voices path not found: {real_path}")
        return False
    
    if os.path.exists(ai_path):
        ai_files = [f for f in os.listdir(ai_path) if f.startswith('ai_') and f.endswith('.wav')]
        print(f"✅ AI voices: {len(ai_files)} files")
    else:
        print(f"❌ AI voices path not found: {ai_path}")
        return False
    
    if len(real_files) > 0 and len(ai_files) > 0:
        print(f"📊 Total: {len(real_files)} real + {len(ai_files)} AI = {len(real_files) + len(ai_files)} samples")
        return True
    else:
        print("❌ No files found!")
        return False

def main():
    parser = argparse.ArgumentParser(description='Launch AASIST Training')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--fast', action='store_true', help='Fast mode (fewer epochs)')
    parser.add_argument('--profile', action='store_true', help='Profile GPU performance')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 AASIST TRAINING LAUNCHER")
    print("=" * 70)
    
    # Check prerequisites
    if not check_gpu():
        response = input("⚠️  No GPU detected. Continue with CPU? (y/n): ")
        if response.lower() != 'y':
            print("❌ Aborting...")
            return
    
    if not check_dataset():
        print("❌ Dataset check failed!")
        return
    
    # Adjust parameters for fast mode
    if args.fast:
        args.epochs = 20
        args.batch_size = 64
        print("\n⚡ Fast mode enabled:")
        print(f"   Epochs: {args.epochs}")
        print(f"   Batch size: {args.batch_size}")
    
    print(f"\n⚙️  Training configuration:")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.lr}")
    
    print("\n" + "=" * 70)
    response = input("🚀 Start training? (y/n): ")
    
    if response.lower() == 'y':
        print("\n▶️  Launching training script...")
        
        # Set environment variables for optimal performance
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        env['CUDA_LAUNCH_BLOCKING'] = '0'
        
        # Run training script
        cmd = [
            sys.executable, 
            "scripts/fine_tune_aasist_high_performance.py"
        ]
        
        try:
            subprocess.run(cmd, env=env, check=True)
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
        except Exception as e:
            print(f"❌ Error running training: {e}")
    else:
        print("❌ Training cancelled")

if __name__ == "__main__":
    main()