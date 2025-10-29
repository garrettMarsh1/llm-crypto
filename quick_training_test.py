#!/usr/bin/env python3
"""
Quick Training Test
Tests the training pipeline with a small subset of data
"""

import os
import json
import torch
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
import yaml
from loguru import logger

def load_training_data(limit=50):
    """Load a small subset of training data for testing"""
    logger.info(f"Loading {limit} samples for quick training test...")
    
    with open('training_data/combined_training_data.json', 'r') as f:
        data = json.load(f)
    
    # Take first N samples
    test_data = data[:limit]
    logger.info(f"Loaded {len(test_data)} samples")
    
    return test_data

def format_training_sample(sample):
    """Format a single training sample into text"""
    price_data = sample['price_data']
    indicators = sample['technical_indicators']
    context = sample['market_context']
    
    # Create a comprehensive prompt
    prompt = """Analyze the following cryptocurrency market data and provide a trading signal:

PRICE DATA:
- Symbol: {symbol}
- Close: ${close:.2f}
- Volume: {volume:.0f}
- High: ${high:.2f}
- Low: ${low:.2f}

TECHNICAL INDICATORS:
- RSI: {rsi:.1f}
- MACD: {macd:.3f}
- MACD Signal: {macd_signal:.3f}
- Bollinger Position: {bb_position:.2f}
- Stochastic K: {stoch_k:.1f}
- Williams %R: {williams_r:.1f}
- Volume Ratio: {volume_ratio:.2f}

MARKET CONTEXT:
- Market Cap: ${market_cap:.0f}
- Fear & Greed: {fear_greed}
- Trend: {trend}
- Volatility: {volatility_regime}
- Volume Regime: {volume_regime}

Based on this analysis, provide your trading recommendation:""".format(
        symbol=sample['symbol'],
        close=price_data['close'],
        volume=price_data['volume'],
        high=price_data['high'],
        low=price_data['low'],
        rsi=indicators['rsi'],
        macd=indicators['macd'],
        macd_signal=indicators['macd_signal'],
        bb_position=indicators['bb_position'],
        stoch_k=indicators['stoch_k'],
        williams_r=indicators['williams_r'],
        volume_ratio=indicators['volume_ratio'],
        market_cap=context['market_cap'],
        fear_greed=context['fear_greed'],
        trend=context['trend'],
        volatility_regime=context['volatility_regime'],
        volume_regime=context['volume_regime']
    )

    # Expected response
    response = """
SIGNAL: {signal}
CONFIDENCE: {confidence:.2f}
REASONING: {reasoning}
POSITION SIZE: {position_size}
STOP LOSS: {stop_loss}
TAKE PROFIT: {take_profit}""".format(
        signal=sample['signal'],
        confidence=sample['confidence'],
        reasoning=sample['reasoning'],
        position_size=sample['position_size'],
        stop_loss=sample['stop_loss'],
        take_profit=sample['take_profit']
    )

    return prompt + response

def prepare_dataset(data):
    """Prepare dataset for training"""
    logger.info("Formatting training samples...")
    
    formatted_samples = []
    for sample in data:
        formatted_text = format_training_sample(sample)
        formatted_samples.append({"text": formatted_text})
    
    # Create dataset
    dataset = Dataset.from_list(formatted_samples)
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    return dataset

def setup_model_and_tokenizer():
    """Setup model and tokenizer for training"""
    logger.info("Setting up model and tokenizer...")
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
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
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    logger.info("Model and tokenizer setup complete")
    return model, tokenizer

def tokenize_function(examples, tokenizer):
    """Tokenize the dataset"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=512
    )

def main():
    """Run quick training test"""
    logger.info("Starting Quick Training Test...")
    
    # Load small dataset
    data = load_training_data(limit=50)
    
    # Prepare dataset
    dataset = prepare_dataset(data)
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer()
    
    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./quick_test_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        learning_rate=2e-4,
        logging_steps=5,
        save_steps=50,
        eval_strategy="no",
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to=None,
        remove_unused_columns=False,
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
    tokenizer.save_pretrained("./quick_test_output")
    
    logger.info("Quick training test completed successfully!")
    
    # Test generation
    logger.info("Testing model generation...")
    test_prompt = "Analyze the following cryptocurrency market data and provide a trading signal:\n\nPRICE DATA:\n- Symbol: BTC-USD\n- Close: $45,000.00\n- Volume: 1000000\n\nTECHNICAL INDICATORS:\n- RSI: 65.5\n- MACD: 0.025\n\nBased on this analysis, provide your trading recommendation:"
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Test generation successful!")
    logger.info(f"Response: {response[len(test_prompt):]}")

if __name__ == "__main__":
    main()
