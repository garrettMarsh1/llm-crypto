#!/usr/bin/env python3
"""
Full Training Pipeline for Crypto Trading Agent
Handles large dataset with streaming and RTX 4070 optimization
"""

import os
import json
import torch
import yaml
import pandas as pd
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, IterableDataset
from loguru import logger
import gc
from typing import Iterator, Dict, Any

def load_config():
    """Load training configuration"""
    with open('training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_training_samples_generator(data_path: str, max_samples: int = 100000) -> Iterator[Dict[str, Any]]:
    """Generator that loads training samples from existing processed data"""
    logger.info(f"Loading training samples from {data_path}...")
    
    # Load existing processed training data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} training samples")
    
    # Yield samples up to max_samples
    count = 0
    for sample in data:
        if count >= max_samples:
            break
        
        yield sample
        count += 1
        
        if count % 1000 == 0:
            logger.info(f"Processed {count} samples...")


def format_training_sample(sample):
    """Format sample for training with strict schema"""
    price_data = sample['price_data']
    indicators = sample['technical_indicators']
    context = sample['market_context']
    
    prompt = f"""Analyze the following cryptocurrency market data and provide a trading signal:

PRICE DATA:
- Symbol: {sample['symbol']}
- Close: ${price_data['close']:.2f}
- Volume: {price_data['volume']:.0f}
- High: ${price_data['high']:.2f}
- Low: ${price_data['low']:.2f}

TECHNICAL INDICATORS:
- RSI: {indicators['rsi']:.1f}
- MACD: {indicators['macd']:.3f}
- MACD Signal: {indicators['macd_signal']:.3f}
- Bollinger Position: {indicators['bb_position']:.2f}

MARKET CONTEXT:
- Market Cap: ${context['market_cap']:.0f}
- Trend: {context['trend']}

Based on this analysis, provide your trading recommendation:"""

    response = f"""
SIGNAL: {sample['signal']}
CONFIDENCE: {sample['confidence']:.2f}
REASONING: {sample['reasoning']}"""

    # Add strict schema
    schema = """
Respond EXACTLY in this format:
SIGNAL: <BUY|SELL|HOLD>
CONFIDENCE: <0.00-1.00>
REASONING: <1-3 concise sentences>."""

    return prompt + response + schema

def create_streaming_dataset(data_path: str, max_samples: int = 100000):
    """Create streaming dataset from processed training data"""
    logger.info(f"Creating streaming dataset with max {max_samples} samples...")
    
    def sample_generator():
        count = 0
        for sample in create_training_samples_generator(data_path, max_samples):
            formatted_text = format_training_sample(sample)
            yield {"text": formatted_text}
            count += 1
            
            if count % 1000 == 0:
                logger.info(f"Generated {count} samples...")
    
    return IterableDataset.from_generator(sample_generator)

def setup_model_and_tokenizer(config):
    """Setup model and tokenizer"""
    logger.info("Setting up model and tokenizer...")
    
    model_name = config['model']['name']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="float16"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype="float16"
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    logger.info("Model setup complete")
    return model, tokenizer

def tokenize_function(examples, tokenizer, max_length=1024):
    """Tokenize examples"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=max_length
    )

def main():
    """Run full training pipeline"""
    logger.info("Starting Full Training Pipeline...")
    
    # Load config
    config = load_config()
    
    # Setup paths
    data_path = "training_data/combined_training_data.json"
    output_dir = "./trained_models/crypto_trading_agent"
    
    # Check if processed data exists
    if not os.path.exists(data_path):
        logger.error(f"Processed training data not found at {data_path}")
        logger.info("Please run the data preparation script first:")
        logger.info("python scripts/prepare_training_data.py --data_dir data_1m")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create streaming dataset
    logger.info("Creating streaming dataset...")
    dataset = create_streaming_dataset(data_path, max_samples=config['data']['max_samples'])
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, config['model']['max_length']),
        batched=True
    )
    
    # Training arguments optimized for RTX 4070
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        warmup_steps=config['training']['warmup_steps'],
        learning_rate=config['training']['learning_rate'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_strategy="no",
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to=None,
        remove_unused_columns=False,
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        max_steps=config['training']['max_steps'],  # Use max_steps for streaming
        save_total_limit=2,
        logging_dir="./logs",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save config
    with open(f"{output_dir}/training_config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    main()
