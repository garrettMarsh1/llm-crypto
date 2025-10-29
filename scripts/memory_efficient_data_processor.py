#!/usr/bin/env python3
"""
Memory-Efficient Data Processor for Large CSV Files
Processes 2.5M+ row datasets without running out of RAM
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import gc
import psutil
import os
from typing import Dict, List, Iterator, Generator
from loguru import logger
import argparse
from datetime import datetime
import yaml

class MemoryEfficientProcessor:
    """Process large CSV files in chunks to avoid memory issues"""
    
    def __init__(self, chunk_size: int = 10000, max_memory_usage: float = 0.8):
        self.chunk_size = chunk_size
        self.max_memory_usage = max_memory_usage
        self.processed_samples = []
        
    def get_memory_usage(self) -> float:
        """Get current memory usage as percentage"""
        return psutil.virtual_memory().percent / 100.0
    
    def log_memory_status(self, stage: str):
        """Log current memory status"""
        memory_percent = self.get_memory_usage() * 100
        available_gb = psutil.virtual_memory().available / (1024**3)
        logger.info(f"{stage} - Memory: {memory_percent:.1f}% used, {available_gb:.1f}GB available")
        
        if memory_percent > 90:
            logger.warning("High memory usage detected! Consider reducing chunk_size")
    
    def should_cleanup_memory(self) -> bool:
        """Check if memory cleanup is needed"""
        return self.get_memory_usage() > self.max_memory_usage
    
    def cleanup_memory(self):
        """Force garbage collection and memory cleanup"""
        gc.collect()
        logger.info("Memory cleanup performed")
    
    def read_csv_in_chunks(self, file_path: Path) -> Iterator[pd.DataFrame]:
        """Read CSV file in memory-efficient chunks"""
        logger.info(f"Reading {file_path} in chunks of {self.chunk_size} rows...")
        
        try:
            chunk_iter = pd.read_csv(
                file_path,
                chunksize=self.chunk_size,
                low_memory=False,
                parse_dates=['timestamp'],
                date_parser=pd.to_datetime
            )
            
            for i, chunk in enumerate(chunk_iter):
                if i % 10 == 0:  # Log every 10 chunks
                    self.log_memory_status(f"Processing chunk {i}")
                
                # Clean up chunk data
                chunk = chunk.dropna()
                chunk = chunk.sort_values('timestamp').reset_index(drop=True)
                
                yield chunk
                
                # Cleanup if memory usage is high
                if self.should_cleanup_memory():
                    self.cleanup_memory()
                    
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise
    
    def calculate_technical_indicators_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators for a chunk of data"""
        
        # Price changes
        df['change_1m'] = df['close'].pct_change() * 100
        df['change_5m'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100).fillna(0)
        df['change_15m'] = ((df['close'] - df['close'].shift(15)) / df['close'].shift(15) * 100).fillna(0)
        df['change_1h'] = ((df['close'] - df['close'].shift(60)) / df['close'].shift(60) * 100).fillna(0)
        df['change_4h'] = ((df['close'] - df['close'].shift(240)) / df['close'].shift(240) * 100).fillna(0)
        df['change_24h'] = ((df['close'] - df['close'].shift(1440)) / df['close'].shift(1440) * 100).fillna(0)
        
        # Moving Averages - ALL periods from original
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
        df['atr'] = self.calculate_atr(df)
        
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
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def generate_samples_from_chunk(self, df: pd.DataFrame, symbol: str, sample_interval: int = 240) -> List[Dict]:
        """Generate training samples from a chunk of data"""
        samples = []
        
        # Start at row 200 to ensure indicators are calculated
        for i in range(200, len(df), sample_interval):
            if i >= len(df):
                break
            
            row = df.iloc[i]
            
            # Skip if critical indicators are NaN
            if pd.isna(row['rsi']) or pd.isna(row['macd']) or pd.isna(row['bb_position']):
                continue
            
            # Create sample with ALL technical indicators from original script
            sample = {
                'symbol': symbol,
                'timestamp': row['timestamp'].isoformat(),
                'price_data': {
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
                },
                'technical_indicators': {
                    'rsi': float(row['rsi']),
                    'macd': float(row['macd']),
                    'macd_signal': float(row['macd_signal']),
                    'macd_histogram': float(row['macd_histogram']),
                    'sma_5': float(row['sma_5']),
                    'sma_10': float(row['sma_10']),
                    'sma_20': float(row['sma_20']),
                    'sma_50': float(row['sma_50']),
                    'sma_100': float(row['sma_100']),
                    'sma_200': float(row['sma_200']),
                    'ema_5': float(row['ema_5']),
                    'ema_10': float(row['ema_10']),
                    'ema_20': float(row['ema_20']),
                    'ema_50': float(row['ema_50']),
                    'ema_100': float(row['ema_100']),
                    'ema_200': float(row['ema_200']),
                    'bb_position': float(row['bb_position']),
                    'bb_width': float(row['bb_width']),
                    'bb_upper': float(row['bb_upper']),
                    'bb_middle': float(row['bb_middle']),
                    'bb_lower': float(row['bb_lower']),
                    'stoch_k': float(row['stoch_k']),
                    'stoch_d': float(row['stoch_d']),
                    'williams_r': float(row['williams_r']),
                    'volume_ratio': float(row['volume_ratio']),
                    'volume_price_trend': float(row['volume_price_trend']),
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
                },
                'market_context': {
                    'market_cap': float(row['close']) * self.get_market_cap_multiplier(symbol),
                    'fear_greed': self.calculate_fear_greed_index(row),
                    'trend': self.determine_trend(row),
                    'volatility_regime': self.determine_volatility_regime(row['volatility']),
                    'volume_regime': self.determine_volume_regime(row['volume_ratio']),
                    'price_momentum': self.calculate_price_momentum(row),
                    'support_resistance': self.calculate_support_resistance_levels(row),
                    'market_structure': self.analyze_market_structure(row)
                },
                'signal': self.generate_advanced_trading_signal(row),
                'confidence': self.calculate_confidence(row),
                'reasoning': self.generate_reasoning(row),
                'position_size': self.calculate_position_size(row),
                'stop_loss': self.calculate_stop_loss(row),
                'take_profit': self.calculate_take_profit(row)
            }
            
            samples.append(sample)
        
        return samples
    
    def get_market_cap_multiplier(self, symbol: str) -> float:
        """Get market cap multiplier for symbol"""
        multipliers = {
            'BTC-USD': 21000000,
            'ETH-USD': 120000000,
            'SOL-USD': 500000000,
        }
        return multipliers.get(symbol, 1000000)
    
    def calculate_fear_greed_index(self, row) -> str:
        """Calculate simplified fear & greed index"""
        rsi = row['rsi']
        bb_pos = row['bb_position']
        
        if rsi < 30 and bb_pos < 0.3:
            return "Extreme Fear"
        elif rsi < 40:
            return "Fear"
        elif rsi > 70 and bb_pos > 0.7:
            return "Extreme Greed"
        elif rsi > 60:
            return "Greed"
        else:
            return "Neutral"
    
    def determine_trend(self, row) -> str:
        """Determine market trend"""
        if row['close'] > row['sma_20'] and row['close'] > row['sma_50']:
            return "Bullish"
        elif row['close'] < row['sma_20'] and row['close'] < row['sma_50']:
            return "Bearish"
        else:
            return "Sideways"
    
    def generate_signal(self, row) -> str:
        """Generate trading signal"""
        rsi = row['rsi']
        macd = row['macd']
        macd_signal = row['macd_signal']
        bb_pos = row['bb_position']
        
        buy_score = 0
        sell_score = 0
        
        # RSI signals
        if rsi < 30:
            buy_score += 2
        elif rsi > 70:
            sell_score += 2
        
        # MACD signals
        if macd > macd_signal:
            buy_score += 1
        else:
            sell_score += 1
        
        # Bollinger Bands
        if bb_pos < 0.2:
            buy_score += 1
        elif bb_pos > 0.8:
            sell_score += 1
        
        if buy_score > sell_score:
            return "BUY"
        elif sell_score > buy_score:
            return "SELL"
        else:
            return "HOLD"
    
    def calculate_confidence(self, row) -> float:
        """Calculate signal confidence"""
        rsi = row['rsi']
        bb_pos = row['bb_position']
        
        # Base confidence
        confidence = 0.5
        
        # RSI confidence boost
        if rsi < 25 or rsi > 75:
            confidence += 0.3
        elif rsi < 35 or rsi > 65:
            confidence += 0.2
        
        # Bollinger Bands confidence
        if bb_pos < 0.1 or bb_pos > 0.9:
            confidence += 0.2
        elif bb_pos < 0.3 or bb_pos > 0.7:
            confidence += 0.1
        
        return min(0.95, confidence)
    
    def calculate_price_momentum(self, row) -> Dict:
        """Calculate price momentum indicators"""
        return {
            'short_term': float(row['change_1h']),
            'medium_term': float(row['change_4h']),
            'long_term': float(row['change_24h']),
            'acceleration': float(row['change_1h']) - float(row['change_4h']),
            'strength': abs(float(row['change_24h']))
        }
    
    def calculate_support_resistance_levels(self, row) -> Dict:
        """Calculate support and resistance levels"""
        current_price = float(row['close'])
        
        return {
            'resistance_1': float(row['bb_upper']),
            'resistance_2': current_price * 1.05,
            'support_1': float(row['bb_lower']),
            'support_2': current_price * 0.95,
            'pivot': float(row['bb_middle']),
            'distance_to_resistance': (float(row['bb_upper']) - current_price) / current_price * 100,
            'distance_to_support': (current_price - float(row['bb_lower'])) / current_price * 100
        }
    
    def analyze_market_structure(self, row) -> Dict:
        """Analyze market structure and patterns"""
        rsi = float(row['rsi'])
        macd = float(row['macd'])
        macd_signal = float(row['macd_signal'])
        bb_position = float(row['bb_position'])
        
        structure = {
            'trend_strength': abs(macd - macd_signal),
            'momentum_direction': 'bullish' if macd > macd_signal else 'bearish',
            'volatility_level': 'high' if float(row['volatility']) > 0.03 else 'low',
            'rsi_regime': 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral',
            'bb_regime': 'upper' if bb_position > 0.8 else 'lower' if bb_position < 0.2 else 'middle',
            'confluence_score': 0
        }
        
        # Calculate confluence score
        confluence = 0
        if structure['momentum_direction'] == 'bullish':
            confluence += 1
        if structure['rsi_regime'] in ['neutral', 'oversold']:
            confluence += 1
        if structure['bb_regime'] in ['lower', 'middle']:
            confluence += 1
        if int(row['trend_short']):
            confluence += 1
        
        structure['confluence_score'] = confluence / 4.0
        return structure
    
    def determine_volatility_regime(self, volatility: float) -> str:
        """Determine volatility regime"""
        if volatility > 0.05:
            return "High Volatility"
        elif volatility > 0.02:
            return "Medium Volatility"
        else:
            return "Low Volatility"
    
    def determine_volume_regime(self, volume_ratio: float) -> str:
        """Determine volume regime"""
        if volume_ratio > 2:
            return "High Volume"
        elif volume_ratio > 1.5:
            return "Above Average Volume"
        elif volume_ratio < 0.5:
            return "Low Volume"
        else:
            return "Normal Volume"
    
    def generate_advanced_trading_signal(self, row) -> str:
        """Generate advanced trading signal with confidence score"""
        rsi = float(row['rsi'])
        macd = float(row['macd'])
        macd_signal = float(row['macd_signal'])
        bb_position = float(row['bb_position'])
        stoch_k = float(row['stoch_k'])
        williams_r = float(row['williams_r'])
        volume_ratio = float(row['volume_ratio'])
        trend_short = int(row['trend_short'])
        trend_medium = int(row['trend_medium'])
        
        buy_score = 0
        sell_score = 0
        
        # RSI analysis
        if rsi < 25:
            buy_score += 3
        elif rsi < 35:
            buy_score += 2
        elif rsi > 75:
            sell_score += 3
        elif rsi > 65:
            sell_score += 2
        
        # MACD analysis
        if macd > macd_signal and macd > 0:
            buy_score += 2
        elif macd < macd_signal and macd < 0:
            sell_score += 2
        elif macd > macd_signal:
            buy_score += 1
        elif macd < macd_signal:
            sell_score += 1
        
        # Bollinger Bands analysis
        if bb_position < 0.1:
            buy_score += 2
        elif bb_position > 0.9:
            sell_score += 2
        elif bb_position < 0.3:
            buy_score += 1
        elif bb_position > 0.7:
            sell_score += 1
        
        # Stochastic analysis
        if stoch_k < 20 and stoch_k > float(row['stoch_d']):
            buy_score += 1
        elif stoch_k > 80 and stoch_k < float(row['stoch_d']):
            sell_score += 1
        
        # Williams %R analysis
        if williams_r < -80:
            buy_score += 1
        elif williams_r > -20:
            sell_score += 1
        
        # Trend analysis
        if trend_short and trend_medium:
            buy_score += 2
        elif not trend_short and not trend_medium:
            sell_score += 2
        
        # Volume confirmation
        if volume_ratio > 1.5:
            if buy_score > sell_score:
                buy_score += 1
            elif sell_score > buy_score:
                sell_score += 1
        
        # Price action patterns
        if int(row['hammer']):
            buy_score += 1
        if int(row['shooting_star']):
            sell_score += 1
        
        # Determine signal
        if buy_score > sell_score:
            return "BUY"
        elif sell_score > buy_score:
            return "SELL"
        else:
            return "HOLD"
    
    def calculate_position_size(self, row) -> str:
        """Calculate recommended position size"""
        confidence = self.calculate_confidence(row)
        volatility = float(row['volatility'])
        
        if confidence > 0.8 and volatility < 0.03:
            return "3-5%"
        elif confidence > 0.7:
            return "2-3%"
        elif confidence > 0.6:
            return "1-2%"
        else:
            return "0.5-1%"
    
    def calculate_stop_loss(self, row) -> str:
        """Calculate recommended stop loss"""
        atr = float(row['atr'])
        volatility = float(row['volatility'])
        close = float(row['close'])
        
        if volatility > 0.05:
            stop_loss_pct = min(5.0, (atr * 2 / close) * 100)
        else:
            stop_loss_pct = min(3.0, (atr * 1.5 / close) * 100)
        
        return f"{stop_loss_pct:.1f}%"
    
    def calculate_take_profit(self, row) -> str:
        """Calculate recommended take profit"""
        volatility = float(row['volatility'])
        confidence = self.calculate_confidence(row)
        
        if confidence > 0.8 and volatility < 0.03:
            return "8-12%"
        elif confidence > 0.7:
            return "5-8%"
        else:
            return "3-5%"
    
    def generate_reasoning(self, row) -> str:
        """Generate trading reasoning"""
        rsi = float(row['rsi'])
        bb_pos = float(row['bb_position'])
        macd = float(row['macd'])
        macd_signal = float(row['macd_signal'])
        
        reasons = []
        
        if rsi < 30:
            reasons.append("RSI indicates oversold conditions")
        elif rsi > 70:
            reasons.append("RSI indicates overbought conditions")
        
        if bb_pos < 0.2:
            reasons.append("Price near lower Bollinger Band")
        elif bb_pos > 0.8:
            reasons.append("Price near upper Bollinger Band")
        
        if macd > macd_signal:
            reasons.append("MACD shows bullish momentum")
        elif macd < macd_signal:
            reasons.append("MACD shows bearish momentum")
        
        if not reasons:
            reasons.append("Mixed technical signals")
        
        return ". ".join(reasons) + "."
    
    def process_file_streaming(self, file_path: Path, symbol: str, output_file: Path, 
                             sample_interval: int = 240, max_samples: int = 5000) -> int:
        """Process a single CSV file in streaming fashion"""
        
        logger.info(f"Processing {file_path} for {symbol}")
        self.log_memory_status("Starting file processing")
        
        total_samples = 0
        chunk_count = 0
        
        try:
            with open(output_file, 'w') as f:
                f.write('[\n')  # Start JSON array
                first_sample = True
                
                for chunk in self.read_csv_in_chunks(file_path):
                    chunk_count += 1
                    logger.info(f"Processing chunk {chunk_count} with {len(chunk)} rows")
                    
                    # Calculate technical indicators
                    chunk = self.calculate_technical_indicators_chunk(chunk)
                    
                    # Generate samples from this chunk
                    samples = self.generate_samples_from_chunk(chunk, symbol, sample_interval)
                    
                    # Write samples to file immediately
                    for sample in samples:
                        if total_samples >= max_samples:
                            break
                        
                        if not first_sample:
                            f.write(',\n')
                        else:
                            first_sample = False
                        
                        json.dump(sample, f, indent=2)
                        total_samples += 1
                    
                    # Cleanup chunk from memory
                    del chunk, samples
                    self.cleanup_memory()
                    
                    if total_samples >= max_samples:
                        break
                
                f.write('\n]')  # End JSON array
            
            logger.info(f"Completed processing {symbol}: {total_samples} samples written")
            return total_samples
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            raise
    
    def process_multiple_files(self, data_dir: str, symbols: List[str], 
                             output_dir: str, **kwargs) -> Dict[str, int]:
        """Process multiple CSV files efficiently"""
        
        results = {}
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for symbol in symbols:
            file_symbol = symbol.replace("-", "_")
            input_file = Path(data_dir) / f"{file_symbol}_1m_last_5y.csv"
            output_file = output_path / f"{file_symbol}_training_data.json"
            
            if not input_file.exists():
                logger.warning(f"File not found: {input_file}")
                continue
            
            try:
                samples_count = self.process_file_streaming(
                    input_file, symbol, output_file, **kwargs
                )
                results[symbol] = samples_count
                
                # Cleanup between files
                self.cleanup_memory()
                
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                results[symbol] = 0
        
        return results

def main():
    """Main function for memory-efficient data processing"""
    
    parser = argparse.ArgumentParser(description="Memory-efficient crypto data processing")
    parser.add_argument("--data_dir", default="./data_1m", help="Input data directory")
    parser.add_argument("--output_dir", default="./training_data", help="Output directory")
    parser.add_argument("--symbols", nargs="+", default=["BTC-USD", "ETH-USD", "SOL-USD"], 
                       help="Crypto symbols to process")
    parser.add_argument("--chunk_size", type=int, default=10000, 
                       help="Number of rows per chunk")
    parser.add_argument("--sample_interval", type=int, default=240, 
                       help="Sample interval in minutes")
    parser.add_argument("--max_samples", type=int, default=5000, 
                       help="Max samples per symbol")
    parser.add_argument("--max_memory", type=float, default=0.8, 
                       help="Max memory usage threshold")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add("logs/memory_efficient_processing_{time}.log")
    logger.info("Starting memory-efficient data processing")
    
    # Initialize processor
    processor = MemoryEfficientProcessor(
        chunk_size=args.chunk_size,
        max_memory_usage=args.max_memory
    )
    
    # Process files
    results = processor.process_multiple_files(
        data_dir=args.data_dir,
        symbols=args.symbols,
        output_dir=args.output_dir,
        sample_interval=args.sample_interval,
        max_samples=args.max_samples
    )
    
    # Print summary
    total_samples = sum(results.values())
    logger.info(f"Processing completed!")
    logger.info(f"Total samples generated: {total_samples}")
    for symbol, count in results.items():
        logger.info(f"  {symbol}: {count} samples")
    
    # Create combined file
    logger.info("Creating combined training data file...")
    combined_samples = []
    
    for symbol in args.symbols:
        file_symbol = symbol.replace("-", "_")
        symbol_file = Path(args.output_dir) / f"{file_symbol}_training_data.json"
        
        if symbol_file.exists():
            with open(symbol_file, 'r') as f:
                samples = json.load(f)
                combined_samples.extend(samples)
    
    combined_file = Path(args.output_dir) / "combined_training_data.json"
    with open(combined_file, 'w') as f:
        json.dump(combined_samples, f, indent=2)
    
    logger.info(f"Combined file created: {combined_file}")
    logger.info(f"Total samples in combined file: {len(combined_samples)}")

if __name__ == "__main__":
    main()
