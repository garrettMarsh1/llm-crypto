#!/usr/bin/env python3
"""
Test Parquet System
Test the Parquet chunk system with a small sample
"""

import json
import tempfile
import shutil
from pathlib import Path
from loguru import logger
import subprocess
import sys

def create_test_data():
    """Create a small test dataset"""
    logger.info("Creating test dataset...")
    
    # Create test data directory
    test_dir = Path("./test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create a small sample of training data
    test_samples = []
    for i in range(100):  # Small dataset for testing
        sample = {
            "symbol": "BTC-USD",
            "timestamp": f"2024-01-01T{i:02d}:00:00Z",
            "price_data": {
                "open": 50000.0 + i * 10,
                "high": 50100.0 + i * 10,
                "low": 49900.0 + i * 10,
                "close": 50050.0 + i * 10,
                "volume": 1000000 + i * 1000,
                "change_1m": 0.1 + i * 0.01,
                "change_5m": 0.5 + i * 0.01,
                "change_15m": 1.0 + i * 0.01,
                "change_1h": 2.0 + i * 0.01,
                "change_4h": 5.0 + i * 0.01,
                "change_24h": 10.0 + i * 0.01
            },
            "technical_indicators": {
                "rsi": 50.0 + i * 0.1,
                "macd": 0.1 + i * 0.01,
                "macd_signal": 0.05 + i * 0.01,
                "macd_histogram": 0.05 + i * 0.01,
                "sma_5": 50000.0 + i * 10,
                "sma_10": 50000.0 + i * 10,
                "sma_20": 50000.0 + i * 10,
                "sma_50": 50000.0 + i * 10,
                "sma_100": 50000.0 + i * 10,
                "sma_200": 50000.0 + i * 10,
                "ema_5": 50000.0 + i * 10,
                "ema_10": 50000.0 + i * 10,
                "ema_20": 50000.0 + i * 10,
                "ema_50": 50000.0 + i * 10,
                "ema_100": 50000.0 + i * 10,
                "ema_200": 50000.0 + i * 10,
                "bb_position": 0.5 + i * 0.01,
                "bb_width": 0.1 + i * 0.001,
                "bb_upper": 51000.0 + i * 10,
                "bb_middle": 50000.0 + i * 10,
                "bb_lower": 49000.0 + i * 10,
                "stoch_k": 50.0 + i * 0.1,
                "stoch_d": 50.0 + i * 0.1,
                "williams_r": -50.0 - i * 0.1,
                "volume_ratio": 1.0 + i * 0.01,
                "volume_price_trend": 1000.0 + i * 10,
                "volatility": 0.02 + i * 0.001,
                "atr": 100.0 + i * 0.1,
                "doji": 0,
                "hammer": 0,
                "shooting_star": 0,
                "trend_short": 1 if i % 2 == 0 else 0,
                "trend_medium": 1 if i % 3 == 0 else 0,
                "trend_long": 1 if i % 5 == 0 else 0,
                "price_vs_resistance": 0.01 + i * 0.001,
                "price_vs_support": -0.01 - i * 0.001
            },
            "market_context": {
                "market_cap": 1000000000000 + i * 1000000,
                "fear_greed": "Neutral",
                "trend": "Sideways",
                "volatility_regime": "Medium Volatility",
                "volume_regime": "Normal Volume",
                "price_momentum": {
                    "short_term": 1.0 + i * 0.01,
                    "medium_term": 2.0 + i * 0.01,
                    "long_term": 5.0 + i * 0.01,
                    "acceleration": 0.1 + i * 0.001,
                    "strength": 5.0 + i * 0.01
                },
                "support_resistance": {
                    "resistance_1": 51000.0 + i * 10,
                    "resistance_2": 52000.0 + i * 10,
                    "support_1": 49000.0 + i * 10,
                    "support_2": 48000.0 + i * 10,
                    "pivot": 50000.0 + i * 10,
                    "distance_to_resistance": 2.0 + i * 0.01,
                    "distance_to_support": -2.0 - i * 0.01
                },
                "market_structure": {
                    "trend_strength": 0.1 + i * 0.001,
                    "momentum_direction": "bullish" if i % 2 == 0 else "bearish",
                    "volatility_level": "low" if i % 3 == 0 else "medium",
                    "rsi_regime": "neutral",
                    "bb_regime": "middle",
                    "confluence_score": 0.5 + i * 0.001
                }
            },
            "signal": "BUY" if i % 3 == 0 else "SELL" if i % 3 == 1 else "HOLD",
            "confidence": 0.5 + (i % 10) * 0.05,
            "reasoning": f"Test reasoning for sample {i}",
            "position_size": "1-2%",
            "stop_loss": "2.0%",
            "take_profit": "5.0%"
        }
        test_samples.append(sample)
    
    # Save test data
    test_file = test_dir / "test_training_data.json"
    with open(test_file, 'w') as f:
        json.dump(test_samples, f, indent=2)
    
    logger.info(f"Created test dataset: {test_file} ({len(test_samples)} samples)")
    return test_file

def test_parquet_conversion(test_file):
    """Test Parquet conversion"""
    logger.info("Testing Parquet conversion...")
    
    cmd = [
        "python", "scripts/parquet_chunk_processor.py",
        "--input_file", str(test_file),
        "--output_dir", "./test_parquet_chunks",
        "--chunk_size", "20",  # Small chunks for testing
        "--max_memory", "0.8"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        logger.info("Parquet conversion test PASSED")
        
        # Check output
        chunks_dir = Path("./test_parquet_chunks")
        if chunks_dir.exists():
            chunk_files = list(chunks_dir.glob("chunk_*.parquet"))
            logger.info(f"Created {len(chunk_files)} Parquet chunks")
            
            # Check metadata
            metadata_file = chunks_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Metadata: {metadata['dataset_info']['total_samples']} samples, "
                           f"{metadata['dataset_info']['total_chunks']} chunks")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Parquet conversion test FAILED: {e}")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        logger.error("Parquet conversion test TIMEOUT")
        return False

def test_parquet_loading():
    """Test loading Parquet chunks"""
    logger.info("Testing Parquet loading...")
    
    try:
        import pyarrow.parquet as pq
        import pandas as pd
        
        chunks_dir = Path("./test_parquet_chunks")
        if not chunks_dir.exists():
            logger.error("No Parquet chunks found for testing")
            return False
        
        chunk_files = list(chunks_dir.glob("chunk_*.parquet"))
        if not chunk_files:
            logger.error("No Parquet files found")
            return False
        
        # Load first chunk
        first_chunk = chunk_files[0]
        table = pq.read_table(first_chunk)
        df = table.to_pandas()
        
        logger.info(f"Loaded chunk: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Sample columns: {list(df.columns)[:10]}")
        
        # Check data integrity
        if 'symbol' in df.columns:
            symbols = df['symbol'].unique()
            logger.info(f"Symbols in chunk: {symbols}")
        
        logger.info("Parquet loading test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Parquet loading test FAILED: {e}")
        return False

def cleanup_test_data():
    """Clean up test data"""
    logger.info("Cleaning up test data...")
    
    test_dirs = ["./test_data", "./test_parquet_chunks"]
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)
            logger.info(f"Removed {test_dir}")

def main():
    """Main test function"""
    
    logger.info("Starting Parquet system test...")
    
    # Check dependencies
    try:
        import pyarrow
        import pandas
        logger.info("Dependencies check PASSED")
    except ImportError as e:
        logger.error(f"Dependencies check FAILED: {e}")
        logger.info("Install with: pip install pyarrow pandas")
        return
    
    try:
        # Create test data
        test_file = create_test_data()
        
        # Test Parquet conversion
        if not test_parquet_conversion(test_file):
            logger.error("Parquet conversion test failed!")
            return
        
        # Test Parquet loading
        if not test_parquet_loading():
            logger.error("Parquet loading test failed!")
            return
        
        logger.info("="*60)
        logger.info("ALL TESTS PASSED!")
        logger.info("="*60)
        logger.info("The Parquet system is working correctly.")
        logger.info("You can now run the full pipeline on your 7.11GB file.")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
    finally:
        # Cleanup
        cleanup_test_data()

if __name__ == "__main__":
    main()
