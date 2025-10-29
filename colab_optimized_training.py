#!/usr/bin/env python3
"""
Colab-optimized training script for A100 GPUs
Run this directly in Google Colab for maximum performance
"""

import os
import sys
import torch
from pathlib import Path

# Add current directory to path
sys.path.append('/content/llm-crypto')

def check_gpu_setup():
    """Check GPU configuration and optimize for A100"""
    print("=== GPU SETUP CHECK ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # Check if it's an A100
        if "A100" in gpu_name:
            print("✅ A100 detected - using optimized settings")
            return "A100"
        elif "V100" in gpu_name:
            print("⚠️  V100 detected - using medium settings")
            return "V100"
        elif "T4" in gpu_name:
            print("⚠️  T4 detected - using conservative settings")
            return "T4"
        else:
            print(f"⚠️  Unknown GPU: {gpu_name} - using default settings")
            return "UNKNOWN"
    else:
        print("❌ No GPU detected!")
        return None

def get_optimized_settings(gpu_type):
    """Get optimized settings based on GPU type"""
    if gpu_type == "A100":
        return {
            "batch_size": 16,
            "gradient_accumulation": 1,
            "num_workers": 8,
            "chunks_per_batch": 4,
            "max_length": 1024,
            "use_bf16": True,
            "use_flash_attention": True
        }
    elif gpu_type == "V100":
        return {
            "batch_size": 8,
            "gradient_accumulation": 2,
            "num_workers": 4,
            "chunks_per_batch": 2,
            "max_length": 512,
            "use_bf16": False,
            "use_flash_attention": False
        }
    elif gpu_type == "T4":
        return {
            "batch_size": 4,
            "gradient_accumulation": 4,
            "num_workers": 2,
            "chunks_per_batch": 1,
            "max_length": 512,
            "use_bf16": False,
            "use_flash_attention": False
        }
    else:
        return {
            "batch_size": 2,
            "gradient_accumulation": 8,
            "num_workers": 1,
            "chunks_per_batch": 1,
            "max_length": 512,
            "use_bf16": False,
            "use_flash_attention": False
        }

def run_optimized_training(data_dir="./parquet_chunks", 
                          output_dir="./trained_model_optimized",
                          max_chunks=5,
                          epochs_per_chunk=1):
    """Run optimized training with GPU-specific settings"""
    
    # Check GPU and get settings
    gpu_type = check_gpu_setup()
    if gpu_type is None:
        print("❌ Cannot run training without GPU!")
        return
    
    settings = get_optimized_settings(gpu_type)
    print(f"\n=== OPTIMIZED SETTINGS FOR {gpu_type} ===")
    for key, value in settings.items():
        print(f"{key}: {value}")
    
    # Import the optimized trainer
    try:
        from scripts.parquet_chunk_trainer_optimized import OptimizedParquetChunkTrainer
        print("✅ Loaded optimized trainer")
    except ImportError:
        print("❌ Could not import optimized trainer - using standard version")
        from scripts.parquet_chunk_trainer import ParquetChunkTrainer as OptimizedParquetChunkTrainer
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize trainer with GPU-specific settings
    trainer = OptimizedParquetChunkTrainer(max_memory_usage=0.90)
    
    print(f"\n=== STARTING TRAINING ===")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max chunks: {max_chunks}")
    print(f"Epochs per chunk: {epochs_per_chunk}")
    
    try:
        # Run training with optimized settings
        if hasattr(trainer, 'train_on_chunks_optimized'):
            trainer.train_on_chunks_optimized(
                data_dir=Path(data_dir),
                output_dir=output_dir,
                epochs_per_chunk=epochs_per_chunk,
                max_chunks=max_chunks,
                chunks_per_batch=settings["chunks_per_batch"]
            )
        else:
            # Fallback to standard training
            trainer.train_on_chunks(
                data_dir=Path(data_dir),
                output_dir=output_dir,
                epochs_per_chunk=epochs_per_chunk,
                max_chunks=max_chunks
            )
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

def monitor_gpu_usage():
    """Monitor GPU usage during training"""
    if torch.cuda.is_available():
        print("\n=== GPU MONITORING ===")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Show utilization if nvidia-ml-py is available
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"GPU Utilization: {util.gpu}%")
            print(f"Memory Utilization: {util.memory}%")
        except ImportError:
            print("Install nvidia-ml-py for detailed GPU monitoring")

if __name__ == "__main__":
    # Run with default settings
    run_optimized_training()
    
    # Monitor GPU after training
    monitor_gpu_usage()
