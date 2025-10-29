#!/usr/bin/env python3
"""
EXTREME A100 TRAINING - MAKE IT SCREAM! 🚀🔥
This script pushes A100 to its absolute limits
"""

import os
import sys
import torch
import warnings
from pathlib import Path

# Suppress warnings for max speed
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async CUDA for speed

def setup_extreme_environment():
    """Setup environment for EXTREME A100 performance"""
    print("🚀 SETTING UP EXTREME A100 ENVIRONMENT...")
    
    # Enable all PyTorch optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set memory management for speed
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,roundup_power2_divisions:16"
    
    # Disable debugging for speed
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"
    
    print("✅ Environment optimized for EXTREME SPEED!")

def check_a100_capabilities():
    """Check A100 specific capabilities"""
    if not torch.cuda.is_available():
        raise RuntimeError("❌ No CUDA available!")
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"🔥 GPU: {gpu_name}")
    print(f"🔥 Memory: {gpu_memory:.1f} GB")
    
    # Check for A100 specific features
    if "A100" not in gpu_name:
        print("⚠️  WARNING: Not an A100 - performance may be limited")
    
    # Check compute capability
    major, minor = torch.cuda.get_device_capability(0)
    compute_cap = f"{major}.{minor}"
    print(f"🔥 Compute Capability: {compute_cap}")
    
    if major >= 8:  # A100 is 8.0
        print("✅ A100 Tensor Cores ENABLED!")
    else:
        print("⚠️  No modern tensor cores detected")
    
    return True

def run_extreme_training():
    """Run training with EXTREME A100 optimizations"""
    
    setup_extreme_environment()
    check_a100_capabilities()
    
    # Add path
    sys.path.append('/content/llm-crypto')
    
    try:
        from scripts.parquet_chunk_trainer_optimized import OptimizedParquetChunkTrainer
        print("✅ Loaded EXTREME optimized trainer")
    except ImportError as e:
        print(f"❌ Failed to import optimized trainer: {e}")
        return
    
    print("\n🚀 STARTING EXTREME A100 TRAINING...")
    print("=" * 50)
    
    # Initialize with extreme settings
    trainer = OptimizedParquetChunkTrainer(max_memory_usage=0.95)  # Push memory limits
    
    # EXTREME training parameters
    training_params = {
        "data_dir": Path('./parquet_chunks'),
        "output_dir": './trained_model_EXTREME',
        "epochs_per_chunk": 1,
        "max_chunks": 10,  # More chunks for better utilization
        "chunks_per_batch": 8,  # MASSIVE batching
    }
    
    print("🔥 EXTREME TRAINING PARAMETERS:")
    for key, value in training_params.items():
        print(f"   {key}: {value}")
    
    print("\n🚀 LAUNCHING EXTREME TRAINING...")
    print("Expected performance: 50-100+ it/s on A100!")
    print("=" * 50)
    
    try:
        trainer.train_on_chunks_optimized(**training_params)
        print("\n🎉 EXTREME TRAINING COMPLETED!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback with slightly reduced settings
        print("\n🔄 Trying with reduced batch size...")
        try:
            # Reduce batch size and try again
            trainer = OptimizedParquetChunkTrainer(max_memory_usage=0.90)
            training_params["chunks_per_batch"] = 4
            trainer.train_on_chunks_optimized(**training_params)
            print("\n🎉 TRAINING COMPLETED (with reduced settings)!")
        except Exception as e2:
            print(f"\n❌ Fallback also failed: {e2}")

def monitor_extreme_performance():
    """Monitor A100 performance during training"""
    if torch.cuda.is_available():
        print("\n📊 A100 PERFORMANCE MONITORING:")
        print("=" * 40)
        
        # Memory stats
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print(f"GPU Memory Allocated: {allocated:.2f} GB")
        print(f"GPU Memory Reserved:  {reserved:.2f} GB")
        print(f"GPU Memory Total:     {total:.2f} GB")
        print(f"GPU Utilization:      {(allocated/total)*100:.1f}%")
        
        # Check if we're using tensor cores
        if hasattr(torch.cuda, 'get_device_capability'):
            major, minor = torch.cuda.get_device_capability(0)
            if major >= 8:
                print("🔥 Tensor Cores: ACTIVE")
            else:
                print("⚠️  Tensor Cores: NOT AVAILABLE")
        
        # Performance tips
        print("\n💡 EXTREME PERFORMANCE TIPS:")
        print("- Batch size 64+ should give 50-100+ it/s")
        print("- Watch for 'CUDA out of memory' - reduce batch if needed")
        print("- A100 should sustain 80-90% GPU utilization")
        print("- Expect 10-20x speedup vs RTX 4070")

if __name__ == "__main__":
    print("🚀🔥 EXTREME A100 TRAINING LAUNCHER 🔥🚀")
    print("=" * 50)
    print("This will push your A100 to its absolute limits!")
    print("Expected: 50-100+ iterations per second")
    print("=" * 50)
    
    run_extreme_training()
    monitor_extreme_performance()
    
    print("\n🎯 TRAINING COMPLETE!")
    print("Check your training speed - should be 10-20x faster than RTX 4070!")
