#!/usr/bin/env python3
"""
Prepare crypto training data from Coinbase CSV files
Converts raw OHLCV data into training-ready format for LLM fine-tuning
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import argparse
from datetime import datetime, timedelta
import yaml
from loguru import logger

def load_and_validate_data(data_dir: str, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Load and validate crypto data from CSV files"""
    
    data = {}
    
    for symbol in symbols:
        file_symbol = symbol.replace("-", "_")
        file_path = Path(data_dir) / f"{file_symbol}_1m_last_5y.csv"
        
        if not file_path.exists():
            logger.warning(f"Data file not found: {file_path}")
            continue
        
        logger.info(f"Loading {symbol} data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Validate data
            if len(df) == 0:
                logger.warning(f"No data found for {symbol}")
                continue
            
            # Check for required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns for {symbol}: {missing_cols}")
                continue
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            # Validate OHLC relationships
            invalid_ohlc = (df['high'] < df['low']) | (df['high'] < df['open']) | (df['high'] < df['close']) | (df['low'] > df['open']) | (df['low'] > df['close'])
            if invalid_ohlc.any():
                logger.warning(f"Found {invalid_ohlc.sum()} invalid OHLC rows for {symbol}, removing them")
                df = df[~invalid_ohlc]
            
            # Validate volume
            negative_volume = df['volume'] < 0
            if negative_volume.any():
                logger.warning(f"Found {negative_volume.sum()} negative volume rows for {symbol}, removing them")
                df = df[~negative_volume]
            
            data[symbol] = df
            logger.info(f"Loaded {len(df)} valid rows for {symbol}")
            
        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")
            continue
    
    return data

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive technical indicators"""
    
    logger.info("Calculating technical indicators...")
    
    # Price changes
    df['change_1m'] = df['close'].pct_change() * 100
    df['change_5m'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100).fillna(0)
    df['change_15m'] = ((df['close'] - df['close'].shift(15)) / df['close'].shift(15) * 100).fillna(0)
    df['change_1h'] = ((df['close'] - df['close'].shift(60)) / df['close'].shift(60) * 100).fillna(0)
    df['change_4h'] = ((df['close'] - df['close'].shift(240)) / df['close'].shift(240) * 100).fillna(0)
    df['change_24h'] = ((df['close'] - df['close'].shift(1440)) / df['close'].shift(1440) * 100).fillna(0)
    
    # Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
    
    # Additional EMAs needed for MACD
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Williams %R
    df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['volume_price_trend'] = (df['volume'] * df['change_1m']).cumsum()
    
    # Volatility indicators
    df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
    df['atr'] = calculate_atr(df)
    
    # Price patterns
    df['doji'] = (abs(df['open'] - df['close']) / (df['high'] - df['low']) < 0.1).astype(int)
    df['hammer'] = ((df['close'] - df['low']) > 2 * (df['open'] - df['close'])) & (df['close'] > df['open']).astype(int)
    df['shooting_star'] = ((df['high'] - df['close']) > 2 * (df['close'] - df['open'])) & (df['close'] < df['open']).astype(int)
    
    # Trend indicators
    df['trend_short'] = (df['close'] > df['sma_20']).astype(int)
    df['trend_medium'] = (df['close'] > df['sma_50']).astype(int)
    df['trend_long'] = (df['close'] > df['sma_200']).astype(int)
    
    # Support and Resistance levels
    df['resistance'] = df['high'].rolling(window=20).max()
    df['support'] = df['low'].rolling(window=20).min()
    df['price_vs_resistance'] = (df['close'] - df['resistance']) / df['resistance']
    df['price_vs_support'] = (df['close'] - df['support']) / df['support']
    
    return df

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=period).mean()
    return atr

def generate_training_samples(df: pd.DataFrame, symbol: str, sample_interval: int = 240) -> List[Dict]:
    """Generate training samples from processed data"""
    
    samples = []
    
    # Sample every 4 hours (240 minutes) to avoid too much data
    for i in range(200, len(df), sample_interval):  # Start at 200 to ensure indicators are calculated
        if i >= len(df):
            break
        
        row = df.iloc[i]
        
        # Skip if any critical indicators are NaN
        if pd.isna(row['rsi']) or pd.isna(row['macd']) or pd.isna(row['bb_position']):
            continue
        
        # Price data
        price_data = {
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume']),
            'change_1m': float(row['change_1m']),
            'change_5m': float(row['change_5m']),
            'change_15m': float(row['change_15m']),
            'change_1h': float(row['change_1h']),
            'change_4h': float(row['change_4h']),
            'change_24h': float(row['change_24h'])
        }
        
        # Technical indicators
        technical_indicators = {
            'rsi': float(row['rsi']),
            'macd': float(row['macd']),
            'macd_signal': float(row['macd_signal']),
            'macd_histogram': float(row['macd_histogram']),
            'sma_20': float(row['sma_20']),
            'sma_50': float(row['sma_50']),
            'sma_200': float(row['sma_200']),
            'bb_position': float(row['bb_position']),
            'bb_width': float(row['bb_width']),
            'stoch_k': float(row['stoch_k']),
            'stoch_d': float(row['stoch_d']),
            'williams_r': float(row['williams_r']),
            'volume_ratio': float(row['volume_ratio']),
            'volatility': float(row['volatility']),
            'atr': float(row['atr']),
            'doji': int(row['doji']),
            'hammer': int(row['hammer']),
            'shooting_star': int(row['shooting_star']),
            'trend_short': int(row['trend_short']),
            'trend_medium': int(row['trend_medium']),
            'trend_long': int(row['trend_long']),
            'price_vs_resistance': float(row['price_vs_resistance']),
            'price_vs_support': float(row['price_vs_support'])
        }
        
        # Market context based on technical analysis and market data
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
        
        # Generate trading signal
        signal, confidence, reasoning = generate_advanced_trading_signal(
            price_data, technical_indicators, market_context
        )
        
        sample = {
            'symbol': symbol,
            'timestamp': row['timestamp'].isoformat(),
            'price_data': price_data,
            'technical_indicators': technical_indicators,
            'market_context': market_context,
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning,
            'position_size': calculate_position_size(technical_indicators, market_context),
            'stop_loss': calculate_stop_loss(price_data, technical_indicators),
            'take_profit': calculate_take_profit(price_data, technical_indicators)
        }
        
        samples.append(sample)
    
    return samples

def get_market_cap_multiplier(symbol: str) -> float:
    """Get approximate market cap multiplier for different cryptocurrencies"""
    multipliers = {
        'BTC-USD': 21000000,  # Approximate BTC supply
        'ETH-USD': 120000000,  # Approximate ETH supply
        'SOL-USD': 500000000,  # Approximate SOL supply
        'DOGE-USD': 140000000000,  # Approximate DOGE supply
    }
    return multipliers.get(symbol, 1000000)

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

def generate_advanced_trading_signal(price_data: Dict, indicators: Dict, context: Dict) -> Tuple[str, float, str]:
    """Generate advanced trading signal with confidence score"""
    
    rsi = indicators['rsi']
    macd = indicators['macd']
    macd_signal = indicators['macd_signal']
    bb_position = indicators['bb_position']
    stoch_k = indicators['stoch_k']
    williams_r = indicators['williams_r']
    volume_ratio = indicators['volume_ratio']
    trend_short = indicators['trend_short']
    trend_medium = indicators['trend_medium']
    
    buy_score = 0
    sell_score = 0
    reasoning_parts = []
    
    # RSI analysis
    if rsi < 25:
        buy_score += 3
        reasoning_parts.append("RSI indicates extreme oversold conditions")
    elif rsi < 35:
        buy_score += 2
        reasoning_parts.append("RSI indicates oversold conditions")
    elif rsi > 75:
        sell_score += 3
        reasoning_parts.append("RSI indicates extreme overbought conditions")
    elif rsi > 65:
        sell_score += 2
        reasoning_parts.append("RSI indicates overbought conditions")
    
    # MACD analysis
    if macd > macd_signal and macd > 0:
        buy_score += 2
        reasoning_parts.append("MACD shows bullish momentum above zero line")
    elif macd < macd_signal and macd < 0:
        sell_score += 2
        reasoning_parts.append("MACD shows bearish momentum below zero line")
    elif macd > macd_signal:
        buy_score += 1
        reasoning_parts.append("MACD shows bullish crossover")
    elif macd < macd_signal:
        sell_score += 1
        reasoning_parts.append("MACD shows bearish crossover")
    
    # Bollinger Bands analysis
    if bb_position < 0.1:
        buy_score += 2
        reasoning_parts.append("Price near lower Bollinger Band (oversold)")
    elif bb_position > 0.9:
        sell_score += 2
        reasoning_parts.append("Price near upper Bollinger Band (overbought)")
    elif bb_position < 0.3:
        buy_score += 1
        reasoning_parts.append("Price in lower half of Bollinger Bands")
    elif bb_position > 0.7:
        sell_score += 1
        reasoning_parts.append("Price in upper half of Bollinger Bands")
    
    # Stochastic analysis
    if stoch_k < 20 and stoch_k > indicators['stoch_d']:
        buy_score += 1
        reasoning_parts.append("Stochastic shows oversold with bullish crossover")
    elif stoch_k > 80 and stoch_k < indicators['stoch_d']:
        sell_score += 1
        reasoning_parts.append("Stochastic shows overbought with bearish crossover")
    
    # Williams %R analysis
    if williams_r < -80:
        buy_score += 1
        reasoning_parts.append("Williams %R indicates oversold conditions")
    elif williams_r > -20:
        sell_score += 1
        reasoning_parts.append("Williams %R indicates overbought conditions")
    
    # Trend analysis
    if trend_short and trend_medium:
        buy_score += 2
        reasoning_parts.append("Strong bullish trend confirmed by moving averages")
    elif not trend_short and not trend_medium:
        sell_score += 2
        reasoning_parts.append("Strong bearish trend confirmed by moving averages")
    
    # Volume confirmation
    if volume_ratio > 1.5:
        if buy_score > sell_score:
            buy_score += 1
            reasoning_parts.append("High volume confirms buying pressure")
        elif sell_score > buy_score:
            sell_score += 1
            reasoning_parts.append("High volume confirms selling pressure")
    
    # Price action patterns
    if indicators['hammer']:
        buy_score += 1
        reasoning_parts.append("Hammer pattern detected (bullish reversal)")
    if indicators['shooting_star']:
        sell_score += 1
        reasoning_parts.append("Shooting star pattern detected (bearish reversal)")
    
    # Determine signal and confidence
    total_signals = buy_score + sell_score
    if total_signals == 0:
        signal = "HOLD"
        confidence = 0.5
        reasoning = "Mixed signals with no clear direction"
    else:
        if buy_score > sell_score:
            signal = "BUY"
            confidence = min(0.95, 0.5 + (buy_score - sell_score) / 10)
        elif sell_score > buy_score:
            signal = "SELL"
            confidence = min(0.95, 0.5 + (sell_score - buy_score) / 10)
        else:
            signal = "HOLD"
            confidence = 0.5
        
        reasoning = ". ".join(reasoning_parts) + "."
    
    return signal, confidence, reasoning

def calculate_position_size(indicators: Dict, context: Dict) -> str:
    """Calculate recommended position size"""
    confidence = indicators.get('confidence', 0.5)
    volatility = indicators.get('volatility', 0.02)
    
    if confidence > 0.8 and volatility < 0.03:
        return "3-5%"
    elif confidence > 0.7:
        return "2-3%"
    elif confidence > 0.6:
        return "1-2%"
    else:
        return "0.5-1%"

def calculate_stop_loss(price_data: Dict, indicators: Dict) -> str:
    """Calculate recommended stop loss"""
    atr = indicators.get('atr', price_data['close'] * 0.02)
    volatility = indicators.get('volatility', 0.02)
    
    if volatility > 0.05:
        stop_loss_pct = min(5.0, (atr * 2 / price_data['close']) * 100)
    else:
        stop_loss_pct = min(3.0, (atr * 1.5 / price_data['close']) * 100)
    
    return f"{stop_loss_pct:.1f}%"

def calculate_take_profit(price_data: Dict, indicators: Dict) -> str:
    """Calculate recommended take profit"""
    volatility = indicators.get('volatility', 0.02)
    confidence = indicators.get('confidence', 0.5)
    
    if confidence > 0.8 and volatility < 0.03:
        return "8-12%"
    elif confidence > 0.7:
        return "5-8%"
    else:
        return "3-5%"

def main():
    parser = argparse.ArgumentParser(description="Prepare crypto training data")
    parser.add_argument("--data_dir", type=str, default="./data_1m", help="Directory containing crypto data")
    parser.add_argument("--symbols", nargs="+", default=["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"], help="Crypto symbols to process")
    parser.add_argument("--output_dir", type=str, default="./training_data", help="Output directory for processed data")
    parser.add_argument("--sample_interval", type=int, default=240, help="Sample interval in minutes (default: 240 = 4 hours)")
    parser.add_argument("--max_samples_per_symbol", type=int, default=5000, help="Maximum samples per symbol")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add("logs/data_preparation_{time}.log")
    logger.info("Starting crypto data preparation for training")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load and validate data
    logger.info("Loading and validating data...")
    data = load_and_validate_data(args.data_dir, args.symbols)
    
    if not data:
        logger.error("No valid data found!")
        return
    
    all_samples = []
    
    # Process each symbol
    for symbol, df in data.items():
        logger.info(f"Processing {symbol} with {len(df)} rows...")
        
        # Calculate technical indicators
        df_processed = calculate_technical_indicators(df)
        
        # Generate training samples
        samples = generate_training_samples(df_processed, symbol, args.sample_interval)
        
        # Limit samples if specified
        if len(samples) > args.max_samples_per_symbol:
            samples = samples[:args.max_samples_per_symbol]
            logger.info(f"Limited {symbol} to {args.max_samples_per_symbol} samples")
        
        all_samples.extend(samples)
        logger.info(f"Generated {len(samples)} samples for {symbol}")
        
        # Save individual symbol data
        symbol_file = output_dir / f"{symbol.replace('-', '_')}_training_data.json"
        with open(symbol_file, 'w') as f:
            json.dump(samples, f, indent=2)
    
    # Save combined data
    combined_file = output_dir / "combined_training_data.json"
    with open(combined_file, 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    # Save metadata
    metadata = {
        "total_samples": len(all_samples),
        "symbols": list(data.keys()),
        "sample_interval_minutes": args.sample_interval,
        "max_samples_per_symbol": args.max_samples_per_symbol,
        "created_at": datetime.now().isoformat(),
        "data_sources": [f"{symbol}_1m_last_5y.csv" for symbol in args.symbols]
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Data preparation completed!")
    logger.info(f"Total samples: {len(all_samples)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Files created:")
    logger.info(f"  - combined_training_data.json")
    logger.info(f"  - metadata.json")
    for symbol in data.keys():
        logger.info(f"  - {symbol.replace('-', '_')}_training_data.json")

if __name__ == "__main__":
    main()
