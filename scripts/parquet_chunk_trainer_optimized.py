#!/usr/bin/env python3
"""
OPTIMIZED Parquet Chunk Trainer for A100 GPUs
High-performance training on Parquet chunks with GPU optimization
Aligned with llm-crypto repository structure
"""

import os
import json
import torch
import gc
import psutil
from pathlib import Path
from typing import Dict, List, Optional
import pyarrow.parquet as pq
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from loguru import logger
import argparse

class OptimizedParquetChunkTrainer:
    """High-performance trainer optimized for A100 GPUs"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
                 max_memory_usage: float = 0.90):  # Higher for A100
        self.model_name = model_name
        self.max_memory_usage = max_memory_usage
        self.tokenizer = None
        self.model = None
        self.metadata = None
        self.trainer = None  # Reuse trainer instance
        
    def get_memory_usage(self) -> float:
        """Get current memory usage as percentage"""
        return psutil.virtual_memory().percent / 100.0
    
    def log_memory_status(self, stage: str) -> None:
        """Log current memory status"""
        memory_percent = self.get_memory_usage() * 100
        available_gb = psutil.virtual_memory().available / (1024**3)

        # GPU memory info
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_cached = torch.cuda.memory_reserved(0) / (1024**3)
            logger.info(f"{stage} - RAM: {memory_percent:.1f}% used, {available_gb:.1f}GB available")
            logger.info(f"{stage} - GPU: {gpu_allocated:.1f}GB/{gpu_memory:.1f}GB allocated, {gpu_cached:.1f}GB cached")
        else:
            logger.info(f"{stage} - Memory: {memory_percent:.1f}% used, {available_gb:.1f}GB available")
    
    def cleanup_memory(self, aggressive: bool = False) -> None:
        """Optimized memory cleanup"""
        if aggressive:
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.synchronize()
    
    def load_metadata(self, data_dir: Path) -> Dict:
        """Load dataset metadata"""
        metadata_file = data_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        logger.info(f"Loaded metadata: {self.metadata['dataset_info']['total_chunks']} chunks, "
                   f"{self.metadata['dataset_info']['total_samples']} samples")
        return self.metadata
    
    def load_parquet_chunk(self, chunk_file: Path) -> pd.DataFrame:
        """Load a single Parquet chunk efficiently"""
        try:
            # Read only needed columns to save memory
            table = pq.read_table(chunk_file)
            df = table.to_pandas()

            # Optimize memory usage
            df = self.optimize_dataframe_memory(df)

            return df
        except Exception as e:
            logger.error(f"Error loading chunk {chunk_file}: {e}")
            raise

    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Convert to category if low cardinality
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')

        return df

    def load_multiple_chunks(self, chunk_files: List[Path], max_chunks_per_batch: int = 3) -> pd.DataFrame:
        """Load multiple chunks at once for better GPU utilization"""
        dfs = []
        for chunk_file in chunk_files[:max_chunks_per_batch]:
            try:
                table = pq.read_table(chunk_file)
                df = table.to_pandas()
                dfs.append(df)
                logger.info(f"Loaded chunk {chunk_file.name} with {len(df)} samples")
            except Exception as e:
                logger.error(f"Error loading chunk {chunk_file}: {e}")
                continue
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Combined {len(dfs)} chunks into {len(combined_df)} samples")
            return combined_df
        else:
            raise ValueError("No chunks could be loaded")
    
    def setup_model_and_tokenizer(self) -> None:
        """Setup model and tokenizer - BALANCED for A100 (fast but stable)"""
        logger.info("Setting up BALANCED model for A100...")
        self.log_memory_status("Before model loading")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # BALANCED quantization - 8bit instead of 4bit for speed/stability
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,  # 8bit instead of 4bit - faster than 4bit, more stable than none
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,  # A100 native
            low_cpu_mem_usage=True,
            use_cache=False,              # Disable KV cache for training speed
        )

        # Prepare for training
        self.model = prepare_model_for_kbit_training(self.model)

        # BALANCED LoRA config
        lora_config = LoraConfig(
            r=8,   # Balanced rank
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)

        self.log_memory_status("After model loading")
        logger.info("BALANCED model setup complete - should be fast and stable!")
    
    def format_training_sample(self, row: pd.Series) -> str:
        """Format a single training sample into text"""
        
        # Extract data from flattened columns
        symbol = row.get('symbol', 'UNKNOWN')
        close = row.get('price_data_close', 0)
        volume = row.get('price_data_volume', 0)
        change_24h = row.get('price_data_change_24h', 0)
        rsi = row.get('technical_indicators_rsi', 50)
        macd = row.get('technical_indicators_macd', 0)
        macd_signal = row.get('technical_indicators_macd_signal', 0)
        bb_position = row.get('technical_indicators_bb_position', 0.5)
        volume_ratio = row.get('technical_indicators_volume_ratio', 1)
        market_cap = row.get('market_context_market_cap', 0)
        fear_greed = row.get('market_context_fear_greed', 'Neutral')
        trend = row.get('market_context_trend', 'Sideways')
        
        prompt = f"""Analyze the following cryptocurrency market data and provide a trading signal:

PRICE DATA:
- Symbol: {symbol}
- Close: ${close:.2f}
- Volume: {volume:.0f}
- Change 24h: {change_24h:+.2f}%

TECHNICAL INDICATORS:
- RSI: {rsi:.1f}
- MACD: {macd:.3f}
- MACD Signal: {macd_signal:.3f}
- Bollinger Position: {bb_position:.2f}
- Volume Ratio: {volume_ratio:.2f}

MARKET CONTEXT:
- Market Cap: ${market_cap:.0f}
- Fear & Greed: {fear_greed}
- Trend: {trend}

Based on this analysis, provide your trading recommendation:"""
        
        response = f"""
SIGNAL: {row.get('signal', 'HOLD')}
CONFIDENCE: {row.get('confidence', 0.5):.2f}
REASONING: {row.get('reasoning', 'No reasoning provided')}"""
        
        schema = (
            "\n\nRespond EXACTLY in this format:\n"
            "SIGNAL: <BUY|SELL|HOLD>\n"
            "CONFIDENCE: <0.00-1.00>\n"
            "REASONING: <1-3 concise sentences>.\n"
        )
        
        return prompt + response + schema
    
    def create_dataset_from_chunk(self, chunk_df: pd.DataFrame) -> Dataset:
        """Create dataset from a Parquet chunk with optimizations"""
        formatted_samples = []
        
        # Vectorized processing for speed
        for idx, row in chunk_df.iterrows():
            try:
                formatted_text = self.format_training_sample(row)
                formatted_samples.append({
                    "text": formatted_text,
                    "sample_id": f"{row.get('symbol', 'UNKNOWN')}_{idx}",
                })
            except Exception as e:
                logger.warning(f"Error formatting sample {idx}: {e}")
                continue
        
        return Dataset.from_list(formatted_samples)
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """EXTREME SPEED tokenization for A100"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",  # Fixed padding for speed
            max_length=512,        # Shorter for SPEED (not 1024)
            return_tensors="pt",
            add_special_tokens=True,
            return_attention_mask=True,
        )
    
    def get_chunk_files(self, data_dir: Path) -> List[Path]:
        """Get list of chunk files in order"""
        chunk_files = sorted(data_dir.glob("chunk_*.parquet"))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in {data_dir}")
        
        logger.info(f"Found {len(chunk_files)} chunk files")
        return chunk_files
    
    def train_on_chunks_optimized(self, data_dir: Path, output_dir: str,
                                 epochs_per_chunk: int = 1, max_chunks: Optional[int] = None,
                                 chunks_per_batch: int = 8) -> None:  # MASSIVE chunk batching
        """Optimized training on Parquet chunks for A100"""
        
        logger.info("Starting OPTIMIZED Parquet chunk training for A100...")
        self.log_memory_status("Before training")
        
        # Load metadata
        self.load_metadata(data_dir)
        
        # Setup model
        self.setup_model_and_tokenizer()
        
        # Get chunk files
        chunk_files = self.get_chunk_files(data_dir)
        
        if max_chunks:
            chunk_files = chunk_files[:max_chunks]
            logger.info(f"Limited to first {max_chunks} chunks")
        
        # EXTREME A100 PERFORMANCE SETTINGS - MAKE IT SCREAM! 🚀
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs_per_chunk,
            per_device_train_batch_size=64,  # MASSIVE batch size for A100!
            gradient_accumulation_steps=1,   # No accumulation needed
            warmup_steps=20,                 # Minimal warmup
            learning_rate=5e-4,              # Higher LR for large batches
            logging_steps=5,                 # Less frequent logging
            save_steps=500,                  # Less frequent saves
            eval_strategy="no",
            save_strategy="steps",
            load_best_model_at_end=False,
            report_to=None,
            remove_unused_columns=False,
            logging_dir="./logs",
            dataloader_pin_memory=True,      # A100 optimization
            dataloader_num_workers=16,       # MAX workers for A100
            bf16=True,                       # A100 native precision
            gradient_checkpointing=False,    # Disabled for max speed
            group_by_length=True,            # Pack sequences efficiently
            ddp_find_unused_parameters=False,
            optim="adamw_torch_fused",       # Fastest optimizer
            max_grad_norm=1.0,
            # EXTREME SPEED OPTIMIZATIONS
            tf32=True,                       # A100 tensor cores
            include_inputs_for_metrics=False,    # Skip unnecessary computations
            prediction_loss_only=True,          # Skip extra loss computations
            # Memory optimizations for larger batches
            max_steps=-1,
            save_total_limit=2,              # Keep fewer checkpoints
            # Additional speed optimizations
            disable_tqdm=False,              # Keep progress bars
            log_level="warning",             # Reduce logging overhead
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Process chunks in batches for better GPU utilization
        total_samples_processed = 0
        batch_count = 0
        
        try:
            for i in range(0, len(chunk_files), chunks_per_batch):
                batch_count += 1
                batch_chunk_files = chunk_files[i:i+chunks_per_batch]
                logger.info(f"Processing batch {batch_count}: {len(batch_chunk_files)} chunks")
                
                # Load multiple chunks at once
                combined_df = self.load_multiple_chunks(batch_chunk_files, chunks_per_batch)
                
                # Create dataset from combined chunks
                dataset = self.create_dataset_from_chunk(combined_df)
                
                # EXTREME SPEED tokenization (CUDA-safe)
                tokenized_dataset = dataset.map(
                    self.tokenize_function,
                    batched=True,
                    batch_size=5000,  # MASSIVE batches for tokenization
                    num_proc=1,       # Single process to avoid CUDA fork issues
                    remove_columns=dataset.column_names,
                    load_from_cache_file=False,  # Skip cache for speed
                    desc="Tokenizing at EXTREME SPEED"
                )
                
                # Create or reuse trainer
                if self.trainer is None:
                    self.trainer = Trainer(
                        model=self.model,
                        args=training_args,
                        train_dataset=tokenized_dataset,
                        data_collator=data_collator,
                    )
                else:
                    # Update dataset for existing trainer
                    self.trainer.train_dataset = tokenized_dataset
                
                # Train on this batch
                logger.info(f"Training on batch {batch_count} with {len(combined_df)} samples...")
                self.trainer.train()
                
                total_samples_processed += len(combined_df)
                logger.info(f"Completed batch {batch_count}. Total samples: {total_samples_processed}")
                
                # Minimal cleanup
                del dataset, tokenized_dataset, combined_df
                self.cleanup_memory(aggressive=False)
                
                # Save checkpoint every few batches
                if batch_count % 3 == 0:
                    checkpoint_dir = Path(output_dir) / f"checkpoint_batch_{batch_count}"
                    self.model.save_pretrained(checkpoint_dir)
                    self.tokenizer.save_pretrained(checkpoint_dir)
                    logger.info(f"Checkpoint saved: {checkpoint_dir}")
            
            # Final save
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Final model saved to: {output_dir}")
            logger.info(f"OPTIMIZED training completed! Total samples: {total_samples_processed}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Final cleanup
            self.cleanup_memory(aggressive=True)

def main() -> None:
    """Main function for optimized Parquet chunk training"""

    parser = argparse.ArgumentParser(description="Train model on Parquet chunks (A100 optimized)")
    parser.add_argument("--data_dir", required=True, help="Directory containing Parquet chunks")
    parser.add_argument("--output_dir", required=True, help="Output directory for trained model")
    parser.add_argument("--epochs_per_chunk", type=int, default=1,
                       help="Number of epochs per chunk")
    parser.add_argument("--max_chunks", type=int, default=None,
                       help="Maximum number of chunks to process")
    parser.add_argument("--chunks_per_batch", type=int, default=3,
                       help="Number of chunks to process together")
    parser.add_argument("--max_memory", type=float, default=0.90,
                       help="Maximum memory usage threshold")

    args = parser.parse_args()

    # Setup logging
    logger.add("logs/optimized_training_{time}.log")
    logger.info("Starting OPTIMIZED Parquet chunk training for A100")

    # Check data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize optimized trainer
    trainer = OptimizedParquetChunkTrainer(max_memory_usage=args.max_memory)

    # Start optimized training
    trainer.train_on_chunks_optimized(
        data_dir=data_dir,
        output_dir=str(output_dir),
        epochs_per_chunk=args.epochs_per_chunk,
        max_chunks=args.max_chunks,
        chunks_per_batch=args.chunks_per_batch
    )

if __name__ == "__main__":
    main()
