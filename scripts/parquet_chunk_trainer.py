#!/usr/bin/env python3
"""
Parquet Chunk Trainer for Large LLM Datasets
Trains models on Parquet chunks without loading entire dataset into memory
"""

import os
import json
import torch
import gc
import psutil
from pathlib import Path
from typing import Iterator, Dict, List, Optional
import pyarrow.parquet as pq
import pyarrow as pa
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
import numpy as np

class ParquetChunkTrainer:
    """Memory-efficient trainer that works with Parquet chunks"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
                 max_memory_usage: float = 0.85):
        self.model_name = model_name
        self.max_memory_usage = max_memory_usage
        self.tokenizer = None
        self.model = None
        self.metadata = None
        
    def get_memory_usage(self) -> float:
        """Get current memory usage as percentage"""
        return psutil.virtual_memory().percent / 100.0
    
    def log_memory_status(self, stage: str):
        """Log current memory status"""
        memory_percent = self.get_memory_usage() * 100
        available_gb = psutil.virtual_memory().available / (1024**3)
        logger.info(f"{stage} - Memory: {memory_percent:.1f}% used, {available_gb:.1f}GB available")
        
        if memory_percent > 90:
            logger.warning("High memory usage detected!")
    
    def cleanup_memory(self):
        """Force garbage collection and memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Memory cleanup performed")
    
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
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with memory optimization"""
        logger.info("Setting up model and tokenizer...")
        self.log_memory_status("Before model loading")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with aggressive quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="float16"
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype="float16",
            low_cpu_mem_usage=True
        )
        
        # Prepare for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA configuration (optimized for memory)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        self.log_memory_status("After model loading")
        logger.info("Model and tokenizer setup complete")
    
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
        """Create dataset from a Parquet chunk"""
        formatted_samples = []
        
        for idx, row in chunk_df.iterrows():
            try:
                formatted_text = self.format_training_sample(row)
                formatted_samples.append({
                    "text": formatted_text,
                    "sample_id": f"{row.get('symbol', 'UNKNOWN')}_{idx}",
                    "original_signal": row.get('signal', 'HOLD'),
                    "original_confidence": row.get('confidence', 0.5)
                })
            except Exception as e:
                logger.warning(f"Error formatting sample {idx}: {e}")
                continue
        
        return Dataset.from_list(formatted_samples)
    
    def tokenize_function(self, examples):
        """Tokenize the dataset"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,  # Optimized for memory
            return_tensors="pt"
        )
    
    def get_chunk_files(self, data_dir: Path) -> List[Path]:
        """Get list of chunk files in order"""
        chunk_files = sorted(data_dir.glob("chunk_*.parquet"))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in {data_dir}")
        
        logger.info(f"Found {len(chunk_files)} chunk files")
        return chunk_files
    
    def train_on_chunks(self, data_dir: Path, output_dir: str, 
                       epochs_per_chunk: int = 1, max_chunks: Optional[int] = None):
        """Train model on Parquet chunks"""
        
        logger.info("Starting Parquet chunk training...")
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
        
        # Training arguments optimized for memory
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs_per_chunk,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,  # Higher for effective batch size
            warmup_steps=2,
            learning_rate=2e-4,
            logging_steps=1,
            save_steps=50,
            eval_strategy="no",
            save_strategy="steps",
            load_best_model_at_end=False,
            report_to=None,
            remove_unused_columns=False,
            logging_dir="./logs",
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            fp16=True,  # Use mixed precision
            gradient_checkpointing=True,  # Save memory
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        total_samples_processed = 0
        chunk_count = 0
        
        try:
            for chunk_file in chunk_files:
                chunk_count += 1
                logger.info(f"Processing chunk {chunk_count}/{len(chunk_files)}: {chunk_file.name}")
                
                # Load chunk
                chunk_df = self.load_parquet_chunk(chunk_file)
                logger.info(f"Loaded chunk with {len(chunk_df)} samples")
                
                # Create dataset from chunk
                dataset = self.create_dataset_from_chunk(chunk_df)
                
                # Tokenize dataset
                tokenized_dataset = dataset.map(
                    self.tokenize_function, 
                    batched=True,
                    remove_columns=dataset.column_names
                )
                
                # Create trainer for this chunk
                trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=tokenized_dataset,
                    data_collator=data_collator,
                )
                
                # Train on this chunk
                logger.info(f"Training on chunk {chunk_count}...")
                trainer.train()
                
                total_samples_processed += len(chunk_df)
                logger.info(f"Completed chunk {chunk_count}. Total samples processed: {total_samples_processed}")
                
                # Cleanup after each chunk
                del dataset, tokenized_dataset, trainer, chunk_df
                self.cleanup_memory()
                
                # Check memory usage
                if self.get_memory_usage() > self.max_memory_usage:
                    logger.warning("Memory usage too high, forcing cleanup...")
                    self.cleanup_memory()
                
                # Save checkpoint every few chunks
                if chunk_count % 5 == 0:
                    checkpoint_dir = Path(output_dir) / f"checkpoint_chunk_{chunk_count}"
                    self.model.save_pretrained(checkpoint_dir)
                    self.tokenizer.save_pretrained(checkpoint_dir)
                    logger.info(f"Checkpoint saved: {checkpoint_dir}")
            
            # Final save
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Final model saved to: {output_dir}")
            logger.info(f"Training completed! Total samples processed: {total_samples_processed}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Final cleanup
            self.cleanup_memory()

def main():
    """Main function for Parquet chunk training"""
    
    parser = argparse.ArgumentParser(description="Train model on Parquet chunks")
    parser.add_argument("--data_dir", required=True, help="Directory containing Parquet chunks")
    parser.add_argument("--output_dir", required=True, help="Output directory for trained model")
    parser.add_argument("--epochs_per_chunk", type=int, default=1, 
                       help="Number of epochs per chunk")
    parser.add_argument("--max_chunks", type=int, default=None, 
                       help="Maximum number of chunks to process")
    parser.add_argument("--max_memory", type=float, default=0.85, 
                       help="Maximum memory usage threshold")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add("logs/parquet_training_{time}.log")
    logger.info("Starting Parquet chunk training")
    
    # Check data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize trainer
    trainer = ParquetChunkTrainer(max_memory_usage=args.max_memory)
    
    # Start training
    trainer.train_on_chunks(
        data_dir=data_dir,
        output_dir=str(output_dir),
        epochs_per_chunk=args.epochs_per_chunk,
        max_chunks=args.max_chunks
    )

if __name__ == "__main__":
    main()
