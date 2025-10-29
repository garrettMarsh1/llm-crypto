"""
Script to train/fine-tune the trading model
"""

import asyncio
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.trading_model import TradingModel, create_sample_training_data
from loguru import logger


async def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Trading Model")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--output-dir", default="./trained_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    
    args = parser.parse_args()
    
    try:
        logger.info("🚀 Starting model training...")
        
        # Initialize model
        model = TradingModel(args.config)
        model.load_model()
        model.setup_lora()
        
        # Create training data
        logger.info("📊 Preparing training data...")
        training_data = create_sample_training_data()
        
        # Add your custom training data here
        # training_data.extend(your_custom_data)
        
        # Prepare dataset
        train_dataset = model.prepare_training_data(training_data)
        logger.info(f"📈 Training dataset size: {len(train_dataset)}")
        
        # Fine-tune model
        logger.info("🎯 Starting fine-tuning...")
        model.fine_tune(
            train_dataset, 
            output_dir=args.output_dir
        )
        
        logger.info(f"✅ Training completed! Model saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
