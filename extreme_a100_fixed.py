#!/usr/bin/env python3
"""
EXTREME A100 TRAINING - FIXED VERSION 🚀🔥
Handles parameter compatibility issues
"""

import os
import sys
import torch
import warnings
from pathlib import Path

# Suppress warnings for max speed
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def setup_extreme_environment():
    """Setup environment for EXTREME A100 performance"""
    print("🚀 SETTING UP EXTREME A100 ENVIRONMENT...")
    
    # Enable all PyTorch optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print("✅ Environment optimized for EXTREME SPEED!")

def patch_training_args_for_extreme_speed():
    """Patch the trainer with extreme A100 settings"""
    
    # Import after path setup
    from scripts.parquet_chunk_trainer_optimized import OptimizedParquetChunkTrainer
    from transformers import TrainingArguments
    
    # Store original method
    original_train = OptimizedParquetChunkTrainer.train_on_chunks_optimized
    
    def extreme_train_on_chunks_optimized(self, data_dir, output_dir, 
                                         epochs_per_chunk=1, max_chunks=None,
                                         chunks_per_batch=8):
        """EXTREME SPEED version with fixed parameters"""
        
        print("🚀 Starting EXTREME A100 training...")
        self.log_memory_status("Before training")
        
        # Load metadata and setup
        self.load_metadata(data_dir)
        self.setup_model_and_tokenizer()
        chunk_files = self.get_chunk_files(data_dir)
        
        if max_chunks:
            chunk_files = chunk_files[:max_chunks]
        
        # EXTREME A100 TRAINING ARGUMENTS - FIXED VERSION
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs_per_chunk,
            per_device_train_batch_size=32,  # Large but safe batch size
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
            dataloader_pin_memory=True,
            dataloader_num_workers=8,  # Reduced to avoid issues
            bf16=True,
            gradient_checkpointing=False,
            group_by_length=True,
            optim="adamw_torch_fused",
            max_grad_norm=1.0,
            tf32=True,
            save_total_limit=2,
            disable_tqdm=False,
            log_level="warning",
        )
        
        # Data collator
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        total_samples_processed = 0
        batch_count = 0
        
        try:
            # Process chunks in large batches
            for i in range(0, len(chunk_files), chunks_per_batch):
                batch_count += 1
                batch_chunk_files = chunk_files[i:i+chunks_per_batch]
                print(f"🔥 Processing EXTREME batch {batch_count}: {len(batch_chunk_files)} chunks")
                
                # Load multiple chunks
                combined_df = self.load_multiple_chunks(batch_chunk_files, chunks_per_batch)
                
                # Create dataset
                dataset = self.create_dataset_from_chunk(combined_df)
                
                # EXTREME SPEED tokenization
                tokenized_dataset = dataset.map(
                    self.tokenize_function, 
                    batched=True,
                    batch_size=2000,  # Large but safe
                    num_proc=8,       # Safe number of processes
                    remove_columns=dataset.column_names,
                    load_from_cache_file=False,
                    desc=f"🚀 EXTREME tokenization batch {batch_count}"
                )
                
                # Create or reuse trainer
                if self.trainer is None:
                    from transformers import Trainer
                    self.trainer = Trainer(
                        model=self.model,
                        args=training_args,
                        train_dataset=tokenized_dataset,
                        data_collator=data_collator,
                    )
                else:
                    self.trainer.train_dataset = tokenized_dataset
                
                # EXTREME TRAINING
                print(f"🔥 EXTREME training on batch {batch_count} with {len(combined_df)} samples...")
                self.trainer.train()
                
                total_samples_processed += len(combined_df)
                print(f"✅ Completed EXTREME batch {batch_count}. Total: {total_samples_processed}")
                
                # Minimal cleanup
                del dataset, tokenized_dataset, combined_df
                self.cleanup_memory(aggressive=False)
                
                # Save checkpoint
                if batch_count % 2 == 0:
                    checkpoint_dir = Path(output_dir) / f"extreme_checkpoint_batch_{batch_count}"
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
            self.cleanup_memory(aggressive=True)
    
    # Apply the patch
    OptimizedParquetChunkTrainer.train_on_chunks_optimized = extreme_train_on_chunks_optimized
    
    return OptimizedParquetChunkTrainer

def run_extreme_training_fixed():
    """Run EXTREME training with fixed parameters"""
    
    setup_extreme_environment()
    
    # Add path
    sys.path.append('/content/llm-crypto')
    
    # Patch the trainer
    try:
        TrainerClass = patch_training_args_for_extreme_speed()
        print("✅ EXTREME trainer patched successfully!")
    except Exception as e:
        print(f"❌ Failed to patch trainer: {e}")
        return
    
    print("\n🚀🔥 STARTING EXTREME A100 TRAINING 🔥🚀")
    print("=" * 60)
    print("Target: 20-50+ iterations per second")
    print("Batch size: 32 (safe but still extreme)")
    print("Chunk batching: 8 chunks at once")
    print("=" * 60)
    
    # Initialize trainer
    trainer = TrainerClass(max_memory_usage=0.90)
    
    try:
        trainer.train_on_chunks_optimized(
            data_dir=Path('./parquet_chunks'),
            output_dir='./trained_model_EXTREME_FIXED',
            epochs_per_chunk=1,
            max_chunks=10,
            chunks_per_batch=8
        )
        
        print("\n🎉 EXTREME TRAINING COMPLETED SUCCESSFULLY!")
        print("Check your training speed - should be 10-20x faster!")
        
    except torch.cuda.OutOfMemoryError:
        print("\n⚠️  GPU OOM - trying with reduced settings...")
        
        # Fallback with smaller batch
        trainer = TrainerClass(max_memory_usage=0.85)
        trainer.train_on_chunks_optimized(
            data_dir=Path('./parquet_chunks'),
            output_dir='./trained_model_EXTREME_FIXED',
            epochs_per_chunk=1,
            max_chunks=5,
            chunks_per_batch=4
        )
        
        print("\n🎉 TRAINING COMPLETED (with reduced settings)!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

def monitor_performance():
    """Monitor A100 performance"""
    if torch.cuda.is_available():
        print("\n📊 A100 PERFORMANCE STATUS:")
        print("=" * 40)
        
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print(f"GPU Memory: {allocated:.1f}GB / {total:.1f}GB ({(allocated/total)*100:.1f}%)")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
        major, minor = torch.cuda.get_device_capability(0)
        print(f"Compute: {major}.{minor} {'(Tensor Cores!)' if major >= 8 else ''}")
        
        print("\n💡 PERFORMANCE EXPECTATIONS:")
        print("- A100 should achieve 20-50+ it/s with batch_size=32")
        print("- Watch GPU utilization - should be 80-90%")
        print("- If still slow, check Colab GPU allocation")

if __name__ == "__main__":
    print("🚀🔥 EXTREME A100 TRAINING - FIXED VERSION 🔥🚀")
    print("This version handles parameter compatibility issues")
    print("=" * 60)
    
    run_extreme_training_fixed()
    monitor_performance()
    
    print("\n🎯 DONE! Your A100 should now be SCREAMING fast! 🚀")
