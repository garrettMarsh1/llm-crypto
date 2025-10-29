#!/usr/bin/env python3
"""
Enhanced Training Test with Full Response Display
Shows complete model responses during training process
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
from loguru import logger
import time

def load_training_data(limit=10):
    """Load a small subset of training data for testing"""
    logger.info(f"Loading {limit} samples for enhanced training test...")
    
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

    # Add strict schema to reduce drift during supervised fine-tuning
    schema = (
        "\n\nRespond EXACTLY in this format:\n"
        "SIGNAL: <BUY|SELL|HOLD>\n"
        "CONFIDENCE: <0.00-1.00>\n"
        "REASONING: <1-3 concise sentences>.\n"
    )

    return prompt + response + schema

def prepare_dataset(data):
    """Prepare dataset for training"""
    logger.info("Formatting training samples...")
    
    formatted_samples = []
    for i, sample in enumerate(data):
        formatted_text = format_training_sample(sample)
        formatted_samples.append({
            "text": formatted_text,
            "sample_id": i,
            "original_signal": sample['signal'],
            "original_confidence": sample['confidence']
        })
    
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
        max_length=1024
    )

class EnhancedTrainer(Trainer):
    """Custom trainer that shows full responses during training"""
    
    def __init__(self, sample_data=None, tokenizer=None, *args, **kwargs):
        # Remove custom parameters before passing to parent
        if 'sample_data' in kwargs:
            kwargs.pop('sample_data')
        if 'tokenizer' in kwargs:
            kwargs.pop('tokenizer')
        
        super().__init__(*args, **kwargs)
        self.sample_data = sample_data or []
        self.tokenizer = tokenizer
    
    def training_step(self, model, inputs):
        """Override training step to show responses"""
        # Get the original training step
        loss = super().training_step(model, inputs)
        
        # Show sample responses every few steps
        if hasattr(self, 'state') and self.state.global_step % 2 == 0:
            self.show_sample_responses(model)
        
        return loss
    
    def show_sample_responses(self, model):
        """Show sample responses from the model"""
        logger.info("\n" + "="*80)
        logger.info("SAMPLE RESPONSES DURING TRAINING")
        logger.info("="*80)
        
        # Test on a few samples
        for i, sample in enumerate(self.sample_data[:3]):
            logger.info(f"\n--- SAMPLE {i+1} ---")
            
            # Create prompt (just the input part)
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
            
            # Expected output
            expected = f"""
SIGNAL: {sample['signal']}
CONFIDENCE: {sample['confidence']:.2f}
REASONING: {sample['reasoning']}"""
            
            logger.info(f"EXPECTED: {expected.strip()}")
            
            # Generate response with schema
            schema = (
                "\n\nRespond EXACTLY in this format:\n"
                "SIGNAL: <BUY|SELL|HOLD>\n"
                "CONFIDENCE: <0.00-1.00>\n"
                "REASONING: <1-3 concise sentences>.\n"
            )
            gen_prompt = prompt + schema
            inputs = self.tokenizer(gen_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.4,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(gen_prompt):].strip()
            
            logger.info(f"GENERATED:")
            logger.info(f"{generated_text}")
            logger.info("-" * 60)

def main():
    """Run enhanced training test with full response display"""
    logger.info("Starting Enhanced Training Test with Full Response Display...")
    
    # Load small dataset
    data = load_training_data(limit=10)
    
    # Prepare dataset
    dataset = prepare_dataset(data)
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer()
    
    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./enhanced_test_output",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        learning_rate=2e-4,
        logging_steps=1,
        save_steps=10,
        eval_strategy="no",
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to=None,
        remove_unused_columns=False,
        logging_dir="./logs",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create enhanced trainer
    trainer = EnhancedTrainer(
        sample_data=data,  # Pass original data for response display
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    # Show initial responses (before training)
    logger.info("\n" + "="*80)
    logger.info("INITIAL RESPONSES (BEFORE TRAINING)")
    logger.info("="*80)
    trainer.show_sample_responses(model)
    
    # Start training
    logger.info("\nStarting training with response monitoring...")
    trainer.train()
    
    # Show final responses (after training)
    logger.info("\n" + "="*80)
    logger.info("FINAL RESPONSES (AFTER TRAINING)")
    logger.info("="*80)
    trainer.show_sample_responses(model)
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained("./enhanced_test_output")
    
    logger.info("Enhanced training test completed successfully!")

if __name__ == "__main__":
    main()
