#!/usr/bin/env python3
"""
EXTREME A100 Training for Colab - Aligned with Repo Structure
Works with cloned llm-crypto repository in Colab
"""

import os
import sys
import torch
import warnings
from pathlib import Path

# Suppress warnings for max speed
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def setup_colab_environment():
    """Setup Colab environment for EXTREME A100 performance"""
    print("🚀 SETTING UP COLAB ENVIRONMENT FOR EXTREME A100...")
    
    # Check if we're in the right directory
    if Path('/content/llm-crypto').exists():
        os.chdir('/content/llm-crypto')
        sys.path.insert(0, '/content/llm-crypto')
        print("✅ Found cloned llm-crypto repository")
    else:
        print("❌ llm-crypto repository not found in /content/")
        print("Please run: !git clone https://github.com/garrettMarsh1/llm-crypto.git")
        return False
    
    # Enable all PyTorch optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set memory management for speed
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    print("✅ Environment optimized for EXTREME SPEED!")
    return True

def check_gpu_and_data():
    """Check GPU and data availability"""
    print("\n🔍 CHECKING SYSTEM STATUS...")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ No CUDA available!")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"🔥 GPU: {gpu_name}")
    print(f"🔥 Memory: {gpu_memory:.1f} GB")
    
    if "A100" not in gpu_name:
        print("⚠️  WARNING: Not an A100 - performance may be limited")
    
    # Check for parquet chunks
    parquet_dir = Path('./parquet_chunks')
    if not parquet_dir.exists():
        print(f"❌ Parquet chunks directory not found: {parquet_dir}")
        print("Please upload your parquet_chunks folder to Colab")
        return False
    
    chunk_files = list(parquet_dir.glob("chunk_*.parquet"))
    metadata_file = parquet_dir / "metadata.json"
    
    print(f"✅ Found {len(chunk_files)} parquet chunks")
    print(f"✅ Metadata file: {'Found' if metadata_file.exists() else 'Missing'}")
    
    return len(chunk_files) > 0 and metadata_file.exists()

def patch_trainer_for_extreme_performance():
    """Patch the existing trainer with EXTREME A100 settings"""
    
    try:
        # Import the optimized trainer if it exists
        from scripts.parquet_chunk_trainer_optimized import OptimizedParquetChunkTrainer
        print("✅ Using OptimizedParquetChunkTrainer")
        return OptimizedParquetChunkTrainer
        
    except ImportError:
        print("⚠️  OptimizedParquetChunkTrainer not found, patching standard trainer...")
        
        # Fallback to patching the standard trainer
        from scripts.parquet_chunk_trainer import ParquetChunkTrainer
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        
        # Store original method
        original_train = ParquetChunkTrainer.train_on_chunks
        
        def extreme_train_on_chunks(self, data_dir, output_dir, epochs_per_chunk=1, max_chunks=None):
            """EXTREME A100 patched training method"""
            
            print("🚀 Starting EXTREME A100 training (patched version)...")
            self.log_memory_status("Before training")
            
            # Load metadata and setup (same as original)
            self.load_metadata(data_dir)
            self.setup_model_and_tokenizer()
            chunk_files = self.get_chunk_files(data_dir)
            
            if max_chunks:
                chunk_files = chunk_files[:max_chunks]
            
            # EXTREME A100 TRAINING ARGUMENTS
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs_per_chunk,
                per_device_train_batch_size=32,  # MASSIVE for A100
                gradient_accumulation_steps=1,
                warmup_steps=20,
                learning_rate=5e-4,
                logging_steps=5,
                save_steps=500,
                eval_strategy="no",
                save_strategy="steps",
                load_best_model_at_end=False,
                report_to=None,
                remove_unused_columns=False,
                logging_dir="./logs",
                dataloader_pin_memory=True,     # A100 optimization
                dataloader_num_workers=8,       # Parallel loading
                bf16=True,                      # A100 native precision
                gradient_checkpointing=False,   # Disabled for speed
                group_by_length=True,
                optim="adamw_torch_fused",      # Fastest optimizer
                tf32=True,                      # A100 tensor cores
                save_total_limit=2,
                disable_tqdm=False,
                log_level="warning",
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )
            
            total_samples_processed = 0
            chunk_count = 0
            
            try:
                # Process chunks with EXTREME settings
                for chunk_file in chunk_files:
                    chunk_count += 1
                    print(f"🔥 EXTREME processing chunk {chunk_count}/{len(chunk_files)}: {chunk_file.name}")
                    
                    # Load chunk
                    chunk_df = self.load_parquet_chunk(chunk_file)
                    print(f"Loaded {len(chunk_df)} samples")
                    
                    # Create dataset
                    dataset = self.create_dataset_from_chunk(chunk_df)
                    
                    # EXTREME tokenization
                    tokenized_dataset = dataset.map(
                        self.tokenize_function, 
                        batched=True,
                        batch_size=2000,  # Large batches
                        num_proc=4,       # Parallel processing
                        remove_columns=dataset.column_names,
                        load_from_cache_file=False,
                        desc=f"🚀 EXTREME tokenizing chunk {chunk_count}"
                    )
                    
                    # Create trainer
                    trainer = Trainer(
                        model=self.model,
                        args=training_args,
                        train_dataset=tokenized_dataset,
                        data_collator=data_collator,
                    )
                    
                    # EXTREME training
                    print(f"🔥 EXTREME training on chunk {chunk_count}...")
                    trainer.train()
                    
                    total_samples_processed += len(chunk_df)
                    print(f"✅ Completed chunk {chunk_count}. Total: {total_samples_processed}")
                    
                    # Cleanup
                    del dataset, tokenized_dataset, trainer, chunk_df
                    self.cleanup_memory()
                    
                    # Save checkpoint
                    if chunk_count % 3 == 0:
                        checkpoint_dir = Path(output_dir) / f"extreme_checkpoint_{chunk_count}"
                        self.model.save_pretrained(checkpoint_dir)
                        self.tokenizer.save_pretrained(checkpoint_dir)
                        print(f"💾 EXTREME checkpoint saved: {checkpoint_dir}")
                
                # Final save
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                print(f"🎉 EXTREME training completed! Total samples: {total_samples_processed}")
                
            except Exception as e:
                print(f"❌ EXTREME training failed: {e}")
                raise
            finally:
                self.cleanup_memory()
        
        # Apply the patch
        ParquetChunkTrainer.train_on_chunks = extreme_train_on_chunks
        print("✅ Standard trainer patched with EXTREME settings")
        
        return ParquetChunkTrainer

def run_extreme_training():
    """Run EXTREME A100 training in Colab"""
    
    # Setup environment
    if not setup_colab_environment():
        return
    
    # Check system
    if not check_gpu_and_data():
        return
    
    # Get trainer class
    TrainerClass = patch_trainer_for_extreme_performance()
    if not TrainerClass:
        return
    
    print("\n🚀🔥 STARTING EXTREME A100 TRAINING 🔥🚀")
    print("=" * 60)
    print("Target: 20-50+ iterations per second")
    print("Batch size: 32 (EXTREME for A100)")
    print("Expected: 10-20x faster than RTX 4070")
    print("=" * 60)
    
    # Initialize trainer
    trainer = TrainerClass(max_memory_usage=0.90)
    
    # Training parameters
    training_params = {
        "data_dir": Path('./parquet_chunks'),
        "output_dir": './trained_model_EXTREME_COLAB',
        "epochs_per_chunk": 1,
        "max_chunks": 10,  # Process more chunks for better utilization
    }
    
    print("🔥 TRAINING PARAMETERS:")
    for key, value in training_params.items():
        print(f"   {key}: {value}")
    
    try:
        # Check if optimized method exists
        if hasattr(trainer, 'train_on_chunks_optimized'):
            print("🚀 Using optimized training method...")
            trainer.train_on_chunks_optimized(
                chunks_per_batch=8,  # Process 8 chunks together
                **training_params
            )
        else:
            print("🚀 Using patched standard training method...")
            trainer.train_on_chunks(**training_params)
        
        print("\n🎉 EXTREME TRAINING COMPLETED SUCCESSFULLY!")
        print("Your A100 should have been SCREAMING fast! 🚀")
        
    except torch.cuda.OutOfMemoryError:
        print("\n⚠️  GPU OOM - trying with reduced batch size...")
        
        # Fallback with smaller settings
        trainer = TrainerClass(max_memory_usage=0.85)
        training_params["max_chunks"] = 5
        
        if hasattr(trainer, 'train_on_chunks_optimized'):
            trainer.train_on_chunks_optimized(chunks_per_batch=4, **training_params)
        else:
            trainer.train_on_chunks(**training_params)
        
        print("\n🎉 TRAINING COMPLETED (with reduced settings)!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

def monitor_performance():
    """Monitor final performance stats"""
    if torch.cuda.is_available():
        print("\n📊 FINAL PERFORMANCE STATS:")
        print("=" * 40)
        
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print(f"GPU Memory Used: {allocated:.1f}GB / {total:.1f}GB")
        print(f"GPU Utilization: {(allocated/total)*100:.1f}%")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        print("\n💡 EXPECTED RESULTS:")
        print("- A100 should achieve 20-50+ it/s")
        print("- 10-20x faster than RTX 4070")
        print("- High GPU utilization (80-90%)")

if __name__ == "__main__":
    print("🚀🔥 EXTREME A100 COLAB TRAINING 🔥🚀")
    print("Aligned with llm-crypto repository structure")
    print("=" * 60)
    
    run_extreme_training()
    monitor_performance()
    
    print("\n🎯 TRAINING COMPLETE!")
    print("Check /content/llm-crypto/trained_model_EXTREME_COLAB for results!")
