#!/usr/bin/env python3
"""
Simple Training Test
Tests basic model loading and generation without full training
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from loguru import logger

def load_sample_data(limit=5):
    """Load a small sample of training data"""
    logger.info(f"Loading {limit} samples for testing...")
    
    with open('training_data/combined_training_data.json', 'r') as f:
        data = json.load(f)
    
    return data[:limit]

def format_sample_prompt(sample):
    """Format a sample into a prompt"""
    price_data = sample['price_data']
    indicators = sample['technical_indicators']
    context = sample['market_context']
    
    prompt = f"""Analyze the following cryptocurrency market data and provide a trading signal:

PRICE DATA:
- Symbol: {sample['symbol']}
- Close: ${price_data['close']:.2f}
- Volume: {price_data['volume']:.0f}

TECHNICAL INDICATORS:
- RSI: {indicators['rsi']:.1f}
- MACD: {indicators['macd']:.3f}
- Bollinger Position: {indicators['bb_position']:.2f}

MARKET CONTEXT:
- Market Cap: ${context['market_cap']:.0f}
- Trend: {context['trend']}

Based on this analysis, provide your trading recommendation:"""
    
    return prompt

def setup_model():
    """Setup model and tokenizer"""
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
    
    logger.info("Model setup complete")
    return model, tokenizer

def test_generation(model, tokenizer, samples):
    """Test model generation on sample data"""
    logger.info("Testing model generation...")
    
    for i, sample in enumerate(samples):
        logger.info(f"\n--- Sample {i+1} ---")
        
        # Format prompt
        prompt = format_sample_prompt(sample)
        logger.info(f"Prompt: {prompt[:200]}...")
        
        # Expected output
        expected = f"""
SIGNAL: {sample['signal']}
CONFIDENCE: {sample['confidence']:.2f}
REASONING: {sample['reasoning']}"""
        
        logger.info(f"Expected: {expected}")
        
        # Add strict schema to the prompt
        schema = (
            "\n\nRespond EXACTLY in this format:\n"
            "SIGNAL: <BUY|SELL|HOLD>\n"
            "CONFIDENCE: <0.00-1.00>\n"
            "REASONING: <1-3 concise sentences>.\n"
        )
        full_prompt = prompt + schema

        # Generate response
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.4,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(full_prompt):].strip()
        
        logger.info(f"Generated: {generated_text}")
        logger.info("-" * 50)

def main():
    """Run simple training test"""
    logger.info("Starting Simple Training Test...")
    
    # Load sample data
    samples = load_sample_data(limit=3)
    
    # Setup model
    model, tokenizer = setup_model()
    
    # Test generation
    test_generation(model, tokenizer, samples)
    
    logger.info("Simple training test completed successfully!")
    logger.info("The model is ready for full training!")

if __name__ == "__main__":
    main()
