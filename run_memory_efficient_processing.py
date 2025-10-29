#!/usr/bin/env python3
"""
Run Memory-Efficient Data Processing
Processes large CSV files without running out of RAM
"""

import subprocess
import sys
from pathlib import Path
from loguru import logger

def run_command(cmd, description):
    """Run a command and handle errors"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Success: {description}")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Error: {e.stderr}")
        return False

def main():
    """Main function to run memory-efficient processing"""
    
    logger.info("Starting memory-efficient data processing pipeline...")
    
    # Check if data directory exists
    data_dir = Path("./data_1m")
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Please ensure your CSV files are in the data_1m directory")
        return
    
    # Check for required CSV files
    required_files = ["BTC_USD_1m_last_5y.csv", "ETH_USD_1m_last_5y.csv", "SOL_USD_1m_last_5y.csv"]
    missing_files = []
    
    for file in required_files:
        if not (data_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"Missing files: {missing_files}")
        logger.info("Processing will continue with available files")
    
    # Step 1: Process data with memory-efficient processor
    logger.info("Step 1: Processing CSV files with memory-efficient processor...")
    
    cmd1 = [
        "python", "scripts/memory_efficient_data_processor.py",
        "--data_dir", "./data_1m",
        "--output_dir", "./training_data",
        "--symbols", "BTC-USD", "ETH-USD", "SOL-USD",
        "--chunk_size", "5000",  # Smaller chunks for safety
        "--sample_interval", "240",  # 4 hours
        "--max_samples", "2000",  # Limit samples per symbol
        "--max_memory", "0.8"  # 80% memory threshold
    ]
    
    if not run_command(cmd1, "Memory-efficient data processing"):
        logger.error("Data processing failed!")
        return
    
    # Step 2: Train model with memory-efficient training
    logger.info("Step 2: Training model with memory-efficient trainer...")
    
    cmd2 = [
        "python", "scripts/memory_efficient_training.py",
        "--data_file", "./training_data/combined_training_data.json",
        "--output_dir", "./memory_efficient_output",
        "--chunk_size", "50",  # Small chunks for training
        "--epochs_per_chunk", "1",
        "--max_memory", "0.8"
    ]
    
    if not run_command(cmd2, "Memory-efficient model training"):
        logger.error("Model training failed!")
        return
    
    logger.info("Memory-efficient processing pipeline completed successfully!")
    logger.info("Check the following directories for results:")
    logger.info("  - training_data/: Processed training data")
    logger.info("  - memory_efficient_output/: Trained model")
    logger.info("  - logs/: Processing logs")

if __name__ == "__main__":
    main()

