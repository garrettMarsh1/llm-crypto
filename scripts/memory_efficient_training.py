#!/usr/bin/env python3
"""
Memory-Efficient Training Script
Trains models on large datasets without running out of RAM
"""

import os
import json
import torch
import gc
import psutil
from pathlib import Path
from typing import Iterator, Dict, List
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

class MemoryEfficientTrainer:
    """Memory-efficient trainer that processes data in chunks"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", 
                 max_memory_usage: float = 0.85):
        self.model_name = model_name
        self.max_memory_usage = max_memory_usage
        self.tokenizer = None
        self.model = None
        
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
        
        # LoRA configuration (smaller for memory efficiency)
        lora_config = LoraConfig(
            r=8,  # Reduced from 16
            lora_alpha=16,  # Reduced from 32
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        self.log_memory_status("After model loading")
        logger.info("Model and tokenizer setup complete")
    
    def load_training_data_chunked(self, data_file: Path, chunk_size: int = 100) -> Iterator[List[Dict]]:
        """Load training data in chunks to avoid memory issues"""
        logger.info(f"Loading training data from {data_file} in chunks of {chunk_size}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        total_samples = len(data)
        logger.info(f"Total samples: {total_samples}")
        
        for i in range(0, total_samples, chunk_size):
            chunk = data[i:i + chunk_size]
            yield chunk
            
            # Cleanup between chunks
            if i % (chunk_size * 5) == 0:  # Every 5 chunks
                self.cleanup_memory()
    
    def format_training_sample(self, sample: Dict) -> str:
        """Format a single training sample into text"""
        price_data = sample['price_data']
        indicators = sample['technical_indicators']
        context = sample['market_context']
        
        prompt = f"""Analyze the following cryptocurrency market data and provide a trading signal:

PRICE DATA:
- Symbol: {sample['symbol']}
- Close: ${price_data['close']:.2f}
- Volume: {price_data['volume']:.0f}
- Change 24h: {price_data['change_24h']:+.2f}%

TECHNICAL INDICATORS:
- RSI: {indicators['rsi']:.1f}
- MACD: {indicators['macd']:.3f}
- MACD Signal: {indicators['macd_signal']:.3f}
- Bollinger Position: {indicators['bb_position']:.2f}
- Volume Ratio: {indicators['volume_ratio']:.2f}

MARKET CONTEXT:
- Market Cap: ${context['market_cap']:.0f}
- Fear & Greed: {context['fear_greed']}
- Trend: {context['trend']}

Based on this analysis, provide your trading recommendation:"""
        
        response = f"""
SIGNAL: {sample['signal']}
CONFIDENCE: {sample['confidence']:.2f}
REASONING: {sample['reasoning']}"""
        
        schema = (
            "\n\nRespond EXACTLY in this format:\n"
            "SIGNAL: <BUY|SELL|HOLD>\n"
            "CONFIDENCE: <0.00-1.00>\n"
            "REASONING: <1-3 concise sentences>.\n"
        )
        
        return prompt + response + schema
    
    def create_dataset_from_chunk(self, chunk: List[Dict]) -> Dataset:
        """Create dataset from a chunk of data"""
        formatted_samples = []
        
        for sample in chunk:
            formatted_text = self.format_training_sample(sample)
            formatted_samples.append({
                "text": formatted_text,
                "sample_id": sample.get('timestamp', 'unknown'),
                "original_signal": sample['signal'],
                "original_confidence": sample['confidence']
            })
        
        return Dataset.from_list(formatted_samples)
    
    def tokenize_function(self, examples):
        """Tokenize the dataset"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512  # Reduced from 1024 for memory efficiency
        )
    
    def train_in_chunks(self, data_file: Path, output_dir: str, 
                       chunk_size: int = 100, epochs_per_chunk: int = 1):
        """Train model in chunks to avoid memory issues"""
        
        logger.info("Starting memory-efficient training...")
        self.log_memory_status("Before training")
        
        # Setup model
        self.setup_model_and_tokenizer()
        
        # Training arguments optimized for memory
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs_per_chunk,
            per_device_train_batch_size=1,  # Keep small
            gradient_accumulation_steps=4,  # Increase to maintain effective batch size
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
            dataloader_pin_memory=False,  # Reduce memory usage
            dataloader_num_workers=0,     # Reduce memory usage
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        chunk_count = 0
        total_samples_processed = 0
        
        try:
            for chunk in self.load_training_data_chunked(data_file, chunk_size):
                chunk_count += 1
                logger.info(f"Processing chunk {chunk_count} with {len(chunk)} samples")
                
                # Create dataset from chunk
                dataset = self.create_dataset_from_chunk(chunk)
                
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
                
                total_samples_processed += len(chunk)
                logger.info(f"Completed chunk {chunk_count}. Total samples processed: {total_samples_processed}")
                
                # Cleanup after each chunk
                del dataset, tokenized_dataset, trainer
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
    """Main function for memory-efficient training"""
    
    parser = argparse.ArgumentParser(description="Memory-efficient model training")
    parser.add_argument("--data_file", default="./training_data/combined_training_data.json", 
                       help="Path to training data file")
    parser.add_argument("--output_dir", default="./memory_efficient_output", 
                       help="Output directory for trained model")
    parser.add_argument("--chunk_size", type=int, default=100, 
                       help="Number of samples per chunk")
    parser.add_argument("--epochs_per_chunk", type=int, default=1, 
                       help="Number of epochs per chunk")
    parser.add_argument("--max_memory", type=float, default=0.85, 
                       help="Maximum memory usage threshold")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add("logs/memory_efficient_training_{time}.log")
    logger.info("Starting memory-efficient training")
    
    # Check if data file exists
    data_file = Path(args.data_file)
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize trainer
    trainer = MemoryEfficientTrainer(max_memory_usage=args.max_memory)
    
    # Start training
    trainer.train_in_chunks(
        data_file=data_file,
        output_dir=str(output_dir),
        chunk_size=args.chunk_size,
        epochs_per_chunk=args.epochs_per_chunk
    )

if __name__ == "__main__":
    main()

