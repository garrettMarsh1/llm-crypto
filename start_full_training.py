#!/usr/bin/env python3
"""
Quick start script for full training pipeline
"""

import os
import sys
from loguru import logger

def main():
    """Start the full training pipeline"""
    logger.info("Starting Full Crypto Trading Agent Training...")
    
    # Check if we're in the right directory
    if not os.path.exists("training_data/combined_training_data.json"):
        logger.error("Training data not found!")
        logger.info("Please run data preparation first:")
        logger.info("python scripts/prepare_training_data.py --data_dir data_1m")
        return
    
    # Check if config exists
    if not os.path.exists("training_config.yaml"):
        logger.error("Training config not found!")
        return
    
    # Start training
    logger.info("Starting training pipeline...")
    os.system("python run_full_training.py")

if __name__ == "__main__":
    main()
