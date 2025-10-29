#!/usr/bin/env python3
"""
Crypto Trading Model Fine-tuning Script
Fine-tunes Qwen2.5-7B on crypto trading data using LoRA/QLoRA
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset
import yaml
from loguru import logger
import warnings
warnings.filterwarnings("ignore")

class CryptoTradingDataset(Dataset):
    """Dataset for crypto trading fine-tuning"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create training prompt
        prompt = self._create_trading_prompt(item)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        inputs["labels"] = inputs["input_ids"].clone()
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": inputs["labels"].squeeze()
        }
    
    def _create_trading_prompt(self, item: Dict) -> str:
        """Create training prompt from market data"""
        
        # Extract data
        symbol = item["symbol"]
        price_data = item["price_data"]
        technical_indicators = item["technical_indicators"]
        market_context = item["market_context"]
        signal = item["signal"]
        reasoning = item["reasoning"]
        
        # Format price data
        price_str = f"Current Price: ${price_data['close']:.2f}\n"
        price_str += f"24h Change: {price_data['change_24h']:.2f}%\n"
        price_str += f"Volume: {price_data['volume_24h']:,.0f}\n"
        
        # Format technical indicators
        tech_str = "Technical Indicators:\n"
        for indicator, value in technical_indicators.items():
            if isinstance(value, float):
                tech_str += f"- {indicator}: {value:.4f}\n"
            else:
                tech_str += f"- {indicator}: {value}\n"
        
        # Format market context
        context_str = "Market Context:\n"
        context_str += f"- Market Cap: ${market_context['market_cap']:,.0f}\n"
        context_str += f"- Fear & Greed Index: {market_context['fear_greed']}\n"
        context_str += f"- Market Trend: {market_context['trend']}\n"
        
        # Create the full prompt
        prompt = f"""<|im_start|>system
You are an expert cryptocurrency trading analyst. Analyze the given market data and provide trading signals with clear reasoning.

<|im_start|>user
Analyze this {symbol} trading data:

{price_str}
{tech_str}
{context_str}

What trading action should I take and why?<|im_end|>

<|im_start|>assistant
Based on my analysis of the {symbol} market data:

**Trading Signal: {signal}**

**Reasoning:**
{reasoning}

**Key Factors:**
- Price Action: {price_data['close']:.2f} ({price_data['change_24h']:+.2f}%)
- Technical Analysis: {technical_indicators.get('rsi', 'N/A')} RSI, {technical_indicators.get('macd_signal', 'N/A')} MACD
- Market Sentiment: {market_context['fear_greed']} ({market_context['trend']} trend)

**Risk Assessment:**
- Position Size: {item.get('position_size', '2-5%')} of portfolio
- Stop Loss: {item.get('stop_loss', '2-3%')} below entry
- Take Profit: {item.get('take_profit', '5-10%')} above entry

This analysis is based on current market conditions and technical indicators. Always consider your risk tolerance and portfolio allocation before trading.<|im_end|>"""

        return prompt

def load_crypto_data(data_dir: str, symbols: List[str]) -> List[Dict]:
    """Load and process crypto data from CSV files"""
    
    all_data = []
    
    for symbol in symbols:
        # Convert symbol format (BTC-USD -> BTC_USD)
        file_symbol = symbol.replace("-", "_")
        file_path = Path(data_dir) / f"{file_symbol}_1m_last_5y.csv"
        
        if not file_path.exists():
            logger.warning(f"Data file not found: {file_path}")
            continue
        
        logger.info(f"Loading data for {symbol} from {file_path}")
        
        # Load CSV data
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Create training samples (every 4 hours to avoid too much data)
        for i in range(240, len(df), 240):  # Every 4 hours (240 minutes)
            if i < len(df):
                sample = create_training_sample(df, i, symbol)
                if sample:
                    all_data.append(sample)
    
    logger.info(f"Loaded {len(all_data)} training samples")
    return all_data

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for the dataframe"""
    
    # Price changes
    df['change_24h'] = ((df['close'] - df['close'].shift(1440)) / df['close'].shift(1440) * 100).fillna(0)
    df['change_1h'] = ((df['close'] - df['close'].shift(60)) / df['close'].shift(60) * 100).fillna(0)
    
    # Moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Volatility
    df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
    
    return df

def create_training_sample(df: pd.DataFrame, idx: int, symbol: str) -> Dict:
    """Create a single training sample from dataframe row"""
    
    if idx < 50:  # Need enough history for indicators
        return None
    
    row = df.iloc[idx]
    
    # Price data
    price_data = {
        'open': float(row['open']),
        'high': float(row['high']),
        'low': float(row['low']),
        'close': float(row['close']),
        'volume': float(row['volume']),
        'change_24h': float(row['change_24h']),
        'change_1h': float(row['change_1h'])
    }
    
    # Technical indicators
    technical_indicators = {
        'rsi': float(row['rsi']) if not pd.isna(row['rsi']) else 50.0,
        'macd': float(row['macd']) if not pd.isna(row['macd']) else 0.0,
        'macd_signal': float(row['macd_signal']) if not pd.isna(row['macd_signal']) else 0.0,
        'sma_20': float(row['sma_20']) if not pd.isna(row['sma_20']) else float(row['close']),
        'sma_50': float(row['sma_50']) if not pd.isna(row['sma_50']) else float(row['close']),
        'bb_position': float(row['bb_position']) if not pd.isna(row['bb_position']) else 0.5,
        'volume_ratio': float(row['volume_ratio']) if not pd.isna(row['volume_ratio']) else 1.0,
        'volatility': float(row['volatility']) if not pd.isna(row['volatility']) else 0.0
    }
    
        # Market context based on technical analysis
        market_context = {
            'market_cap': price_data['close'] * get_market_cap_multiplier(symbol),
            'fear_greed': calculate_fear_greed_index(technical_indicators),
            'trend': determine_market_trend(technical_indicators),
            'volatility_regime': determine_volatility_regime(technical_indicators['volatility']),
            'volume_regime': determine_volume_regime(technical_indicators['volume_ratio']),
            'price_momentum': calculate_price_momentum(price_data),
            'support_resistance': calculate_support_resistance_levels(price_data, technical_indicators),
            'market_structure': analyze_market_structure(technical_indicators)
        }
    
    # Generate trading signal based on technical analysis
    signal, reasoning = generate_trading_signal(price_data, technical_indicators, market_context)
    
    return {
        'symbol': symbol,
        'price_data': price_data,
        'technical_indicators': technical_indicators,
        'market_context': market_context,
        'signal': signal,
        'reasoning': reasoning,
        'position_size': '2-5%',
        'stop_loss': '2-3%',
        'take_profit': '5-10%'
    }

def get_market_cap_multiplier(symbol: str) -> float:
    """Get approximate market cap multiplier for different cryptocurrencies"""
    multipliers = {
        'BTC-USD': 21000000,  # Approximate BTC supply
        'ETH-USD': 120000000,  # Approximate ETH supply
        'SOL-USD': 500000000,  # Approximate SOL supply
        'DOGE-USD': 140000000000,  # Approximate DOGE supply
    }
    return multipliers.get(symbol, 1000000)

def calculate_fear_greed_index(indicators: Dict) -> str:
    """Calculate comprehensive fear & greed index based on technical indicators"""
    rsi = indicators['rsi']
    bb_position = indicators['bb_position']
    volume_ratio = indicators['volume_ratio']
    
    score = 0
    
    # RSI component
    if rsi < 20:
        score += 3
    elif rsi < 30:
        score += 2
    elif rsi > 80:
        score -= 3
    elif rsi > 70:
        score -= 2
    
    # Bollinger Bands component
    if bb_position < 0.1:
        score += 2
    elif bb_position > 0.9:
        score -= 2
    
    # Volume component
    if volume_ratio > 2:
        score += 1
    elif volume_ratio < 0.5:
        score -= 1
    
    if score >= 3:
        return "Extreme Greed"
    elif score >= 1:
        return "Greed"
    elif score <= -3:
        return "Extreme Fear"
    elif score <= -1:
        return "Fear"
    else:
        return "Neutral"

def determine_market_trend(indicators: Dict) -> str:
    """Determine overall market trend"""
    trend_short = indicators['trend_short']
    trend_medium = indicators['trend_medium']
    trend_long = indicators['trend_long']
    
    if trend_short and trend_medium and trend_long:
        return "Strong Bullish"
    elif trend_short and trend_medium:
        return "Bullish"
    elif not trend_short and not trend_medium and not trend_long:
        return "Strong Bearish"
    elif not trend_short and not trend_medium:
        return "Bearish"
    else:
        return "Sideways"

def determine_volatility_regime(volatility: float) -> str:
    """Determine volatility regime"""
    if volatility > 0.05:
        return "High Volatility"
    elif volatility > 0.02:
        return "Medium Volatility"
    else:
        return "Low Volatility"

def determine_volume_regime(volume_ratio: float) -> str:
    """Determine volume regime"""
    if volume_ratio > 2:
        return "High Volume"
    elif volume_ratio > 1.5:
        return "Above Average Volume"
    elif volume_ratio < 0.5:
        return "Low Volume"
    else:
        return "Normal Volume"

def calculate_price_momentum(price_data: Dict) -> Dict:
    """Calculate price momentum indicators"""
    return {
        'short_term': price_data['change_1h'],
        'medium_term': price_data['change_4h'],
        'long_term': price_data['change_24h'],
        'acceleration': price_data['change_1h'] - price_data['change_4h'],
        'strength': abs(price_data['change_24h'])
    }

def calculate_support_resistance_levels(price_data: Dict, indicators: Dict) -> Dict:
    """Calculate support and resistance levels"""
    current_price = price_data['close']
    
    # Use Bollinger Bands as dynamic support/resistance
    bb_upper = indicators.get('bb_upper', current_price * 1.02)
    bb_lower = indicators.get('bb_lower', current_price * 0.98)
    bb_middle = indicators.get('bb_middle', current_price)
    
    return {
        'resistance_1': bb_upper,
        'resistance_2': current_price * 1.05,  # 5% above current
        'support_1': bb_lower,
        'support_2': current_price * 0.95,     # 5% below current
        'pivot': bb_middle,
        'distance_to_resistance': (bb_upper - current_price) / current_price * 100,
        'distance_to_support': (current_price - bb_lower) / current_price * 100
    }

def analyze_market_structure(indicators: Dict) -> Dict:
    """Analyze market structure and patterns"""
    rsi = indicators['rsi']
    macd = indicators['macd']
    macd_signal = indicators['macd_signal']
    bb_position = indicators['bb_position']
    
    # Market structure analysis
    structure = {
        'trend_strength': abs(macd - macd_signal),
        'momentum_direction': 'bullish' if macd > macd_signal else 'bearish',
        'volatility_level': 'high' if indicators['volatility'] > 0.03 else 'low',
        'rsi_regime': 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral',
        'bb_regime': 'upper' if bb_position > 0.8 else 'lower' if bb_position < 0.2 else 'middle',
        'confluence_score': 0  # Will be calculated below
    }
    
    # Calculate confluence score (how many indicators agree)
    confluence = 0
    if structure['momentum_direction'] == 'bullish':
        confluence += 1
    if structure['rsi_regime'] in ['neutral', 'oversold']:
        confluence += 1
    if structure['bb_regime'] in ['lower', 'middle']:
        confluence += 1
    if indicators['trend_short']:
        confluence += 1
    
    structure['confluence_score'] = confluence / 4.0  # Normalize to 0-1
    
    return structure

def generate_trading_signal(price_data: Dict, technical_indicators: Dict, market_context: Dict) -> Tuple[str, str]:
    """Generate trading signal based on technical analysis"""
    
    rsi = technical_indicators['rsi']
    macd = technical_indicators['macd']
    macd_signal = technical_indicators['macd_signal']
    bb_position = technical_indicators['bb_position']
    volume_ratio = technical_indicators['volume_ratio']
    change_24h = price_data['change_24h']
    
    # Simple signal generation logic
    buy_signals = 0
    sell_signals = 0
    reasoning_parts = []
    
    # RSI signals
    if rsi < 30:
        buy_signals += 2
        reasoning_parts.append("RSI indicates oversold conditions")
    elif rsi > 70:
        sell_signals += 2
        reasoning_parts.append("RSI indicates overbought conditions")
    
    # MACD signals
    if macd > macd_signal and macd > 0:
        buy_signals += 1
        reasoning_parts.append("MACD shows bullish momentum")
    elif macd < macd_signal and macd < 0:
        sell_signals += 1
        reasoning_parts.append("MACD shows bearish momentum")
    
    # Bollinger Bands
    if bb_position < 0.2:
        buy_signals += 1
        reasoning_parts.append("Price near lower Bollinger Band")
    elif bb_position > 0.8:
        sell_signals += 1
        reasoning_parts.append("Price near upper Bollinger Band")
    
    # Volume confirmation
    if volume_ratio > 1.5:
        if buy_signals > sell_signals:
            buy_signals += 1
            reasoning_parts.append("High volume confirms buying pressure")
        elif sell_signals > buy_signals:
            sell_signals += 1
            reasoning_parts.append("High volume confirms selling pressure")
    
    # 24h trend
    if change_24h > 5:
        buy_signals += 1
        reasoning_parts.append("Strong 24h uptrend")
    elif change_24h < -5:
        sell_signals += 1
        reasoning_parts.append("Strong 24h downtrend")
    
    # Determine final signal
    if buy_signals > sell_signals and buy_signals >= 3:
        signal = "BUY"
    elif sell_signals > buy_signals and sell_signals >= 3:
        signal = "SELL"
    else:
        signal = "HOLD"
    
    # Create reasoning
    if not reasoning_parts:
        reasoning_parts = ["Mixed signals, maintaining current position"]
    
    reasoning = ". ".join(reasoning_parts) + "."
    
    return signal, reasoning

def setup_model_and_tokenizer(model_name: str, config: Dict):
    """Setup model and tokenizer for training"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune crypto trading model")
    parser.add_argument("--data_dir", type=str, default="./data_1m", help="Directory containing crypto data")
    parser.add_argument("--symbols", nargs="+", default=["BTC-USD", "ETH-USD", "SOL-USD"], help="Crypto symbols to train on")
    parser.add_argument("--output_dir", type=str, default="./trained_models/crypto_trading", help="Output directory for trained model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum training samples")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test split ratio")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger.add("logs/training_{time}.log")
    logger.info("Starting crypto trading model fine-tuning")
    
    # Load data
    logger.info("Loading crypto data...")
    training_data = load_crypto_data(args.data_dir, args.symbols)
    
    if len(training_data) == 0:
        logger.error("No training data found!")
        return
    
    # Limit samples if specified
    if len(training_data) > args.max_samples:
        training_data = training_data[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")
    
    # Split data
    split_idx = int(len(training_data) * (1 - args.test_split))
    train_data = training_data[:split_idx]
    test_data = training_data[split_idx:]
    
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Test samples: {len(test_data)}")
    
    # Setup model
    logger.info("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(
        config['model']['name'], 
        config['training']
    )
    
    # Create datasets
    train_dataset = CryptoTradingDataset(train_data, tokenizer)
    test_dataset = CryptoTradingDataset(test_data, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        num_train_epochs=config['training']['epochs'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training config
    with open(os.path.join(args.output_dir, "training_config.json"), 'w') as f:
        json.dump({
            "model_name": config['model']['name'],
            "symbols": args.symbols,
            "training_samples": len(train_data),
            "test_samples": len(test_data),
            "training_args": training_args.to_dict()
        }, f, indent=2)
    
    logger.info(f"Training completed! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
