#!/usr/bin/env python3
"""
Quick Start Script for Crypto Trading Model Training
Simplified interface for training your crypto trading LLM
"""

import os
import sys
import subprocess
from pathlib import Path
from loguru import logger

def print_banner():
    """Print training banner"""
    print("=" * 70)
    print("🚀 CRYPTO TRADING MODEL TRAINING")
    print("=" * 70)
    print("Fine-tune Qwen2.5-7B on your crypto trading data")
    print("=" * 70)

def check_data_files():
    """Check if data files exist"""
    data_dir = Path("./data_1m")
    required_files = [
        "BTC_USD_1m_last_5y.csv",
        "ETH_USD_1m_last_5y.csv",
        "SOL_USD_1m_last_5y.csv",
        "DOGE_USD_1m_last_5y.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not (data_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n📥 Download data first:")
        print("   python data_for_training.py")
        return False
    
    print("✅ All data files found")
    return True

def check_gpu_memory():
    """Check GPU memory availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU detected: {gpu_memory:.1f} GB VRAM")
            
            if gpu_memory < 8:
                print("⚠️  Warning: Less than 8GB VRAM detected")
                print("   Consider reducing batch_size in training_config.yaml")
            
            return True
        else:
            print("⚠️  No GPU detected, training will be slower on CPU")
            return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def estimate_training_time():
    """Estimate training time"""
    print("\n⏱️  Training Time Estimates:")
    print("   - Data preparation: 5-15 minutes")
    print("   - Model training: 2-6 hours (depending on GPU)")
    print("   - Model evaluation: 10-30 minutes")
    print("   - Total: 3-7 hours")
    print("\n💡 Tip: You can stop and resume training anytime")

def run_training():
    """Run the training pipeline"""
    print("\n🚀 Starting training pipeline...")
    
    cmd = [
        sys.executable, "scripts/run_training_pipeline.py",
        "--config", "training_config.yaml",
        "--data_dir", "./data_1m",
        "--output_dir", "./trained_models"
    ]
    
    try:
        # Run training pipeline
        result = subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        print("Check logs/ directory for detailed error messages")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return False

def show_next_steps():
    """Show next steps after training"""
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETED!")
    print("=" * 70)
    print("\n📁 Your trained model is located at:")
    print("   ./trained_models/trained_model/")
    print("\n🔧 To use your trained model:")
    print("   1. Update config.yaml:")
    print("      model_path: ./trained_models/trained_model")
    print("\n   2. Test the model:")
    print("      python -c \"from scripts.test_system import test_system; import asyncio; asyncio.run(test_system())\"")
    print("\n   3. Start trading:")
    print("      python main.py --mode paper --symbols BTC/USD")
    print("\n   4. Monitor performance:")
    print("      python scripts/monitor_dashboard.py")
    print("\n📊 Training data and logs:")
    print("   - Training data: ./trained_models/training_data/")
    print("   - Logs: ./logs/")
    print("   - Model checkpoints: ./trained_models/trained_model/")

def main():
    """Main training interface"""
    print_banner()
    
    # Check prerequisites
    print("\n🔍 Checking prerequisites...")
    
    if not check_data_files():
        return 1
    
    if not check_gpu_memory():
        return 1
    
    # Show training estimates
    estimate_training_time()
    
    # Confirm training
    print("\n❓ Ready to start training?")
    print("   This will:")
    print("   - Process your crypto data")
    print("   - Fine-tune the Qwen2.5-7B model")
    print("   - Create a specialized crypto trading model")
    
    response = input("\n   Continue? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Training cancelled")
        return 0
    
    # Run training
    success = run_training()
    
    if success:
        show_next_steps()
        return 0
    else:
        print("\n❌ Training failed. Check the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
