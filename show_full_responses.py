#!/usr/bin/env python3
"""
Show Full Model Responses
Demonstrates complete model responses without training
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

def load_sample_data(limit=3):
    """Load sample data for testing"""
    logger.info(f"Loading {limit} samples for full response demonstration...")
    
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
- High: ${price_data['high']:.2f}
- Low: ${price_data['low']:.2f}

TECHNICAL INDICATORS:
- RSI: {indicators['rsi']:.1f}
- MACD: {indicators['macd']:.3f}
- MACD Signal: {indicators['macd_signal']:.3f}
- Bollinger Position: {indicators['bb_position']:.2f}
- Stochastic K: {indicators['stoch_k']:.1f}
- Williams %R: {indicators['williams_r']:.1f}
- Volume Ratio: {indicators['volume_ratio']:.2f}

MARKET CONTEXT:
- Market Cap: ${context['market_cap']:.0f}
- Fear & Greed: {context['fear_greed']}
- Trend: {context['trend']}
- Volatility: {context['volatility_regime']}
- Volume Regime: {context['volume_regime']}

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

def show_full_responses(model, tokenizer, samples):
    """Show complete model responses"""
    logger.info("Showing FULL model responses...")
    
    for i, sample in enumerate(samples):
        print("\n" + "="*100)
        print(f"SAMPLE {i+1} - COMPLETE RESPONSE")
        print("="*100)
        
        # Format prompt and enforce strict schema
        prompt = format_sample_prompt(sample)
        schema = (
            "\n\nRespond EXACTLY in this format:\n"
            "SIGNAL: <BUY|SELL|HOLD>\n"
            "CONFIDENCE: <0.00-1.00>\n"
            "REASONING: <1-3 concise sentences>.\n"
        )
        full_prompt = prompt + schema
        print(f"\nPROMPT:")
        print("-" * 50)
        print(full_prompt)
        
        # Expected output
        expected = f"""
SIGNAL: {sample['signal']}
CONFIDENCE: {sample['confidence']:.2f}
REASONING: {sample['reasoning']}
POSITION SIZE: {sample['position_size']}
STOP LOSS: {sample['stop_loss']}
TAKE PROFIT: {sample['take_profit']}"""
        
        print(f"\nEXPECTED OUTPUT:")
        print("-" * 50)
        print(expected.strip())
        
        # Generate response
        print(f"\nMODEL GENERATED OUTPUT:")
        print("-" * 50)
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=600,
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
        
        print(generated_text)
        print("\n" + "="*100)

def main():
    """Run full response demonstration"""
    logger.info("Starting Full Response Demonstration...")
    
    # Load sample data
    samples = load_sample_data(limit=3)
    
    # Setup model
    model, tokenizer = setup_model()
    
    # Show full responses
    show_full_responses(model, tokenizer, samples)
    
    logger.info("Full response demonstration completed!")

if __name__ == "__main__":
    main()
