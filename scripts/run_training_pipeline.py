#!/usr/bin/env python3
"""
Complete Crypto Trading Model Training Pipeline
Orchestrates data preparation, model training, and evaluation
"""

import os
import sys
import subprocess
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from loguru import logger
import json

def setup_logging():
    """Setup logging for the training pipeline"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        log_dir / "training_pipeline_{time}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="100 MB"
    )

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        "torch", "transformers", "peft", "datasets", "accelerate",
        "bitsandbytes", "pandas", "numpy", "scikit-learn", "matplotlib",
        "seaborn", "plotly", "wandb", "tensorboard"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.info("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_data_availability(data_dir: str, symbols: list) -> bool:
    """Check if required data files exist"""
    missing_files = []
    
    for symbol in symbols:
        file_symbol = symbol.replace("-", "_")
        file_path = Path(data_dir) / f"{file_symbol}_1m_last_5y.csv"
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        logger.error("Missing data files:")
        for file in missing_files:
            logger.error(f"  - {file}")
        logger.info("Run data_for_training.py first to download the data")
        return False
    
    return True

def run_data_preparation(config: dict, args) -> bool:
    """Run data preparation step"""
    logger.info("=" * 60)
    logger.info("STEP 1: DATA PREPARATION")
    logger.info("=" * 60)
    
    cmd = [
        "python", "scripts/prepare_training_data.py",
        "--data_dir", args.data_dir,
        "--symbols"] + config['crypto']['symbols'] + [
        "--output_dir", args.output_dir + "/training_data",
        "--sample_interval", str(config['data']['sample_interval']),
        "--max_samples_per_symbol", str(config['data']['max_samples'])
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Data preparation completed successfully")
        logger.debug(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Data preparation failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def run_model_training(config: dict, args) -> bool:
    """Run model training step"""
    logger.info("=" * 60)
    logger.info("STEP 2: MODEL TRAINING")
    logger.info("=" * 60)
    
    cmd = [
        "python", "scripts/train_crypto_model.py",
        "--data_dir", args.data_dir,
        "--symbols"] + config['crypto']['symbols'] + [
        "--output_dir", args.output_dir + "/trained_model",
        "--config", args.config,
        "--max_samples", str(config['data']['max_samples']),
        "--test_split", str(config['data']['test_split'])
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Model training completed successfully")
        logger.debug(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Model training failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def run_model_evaluation(config: dict, args) -> bool:
    """Run model evaluation step"""
    logger.info("=" * 60)
    logger.info("STEP 3: MODEL EVALUATION")
    logger.info("=" * 60)
    
    # Create evaluation script if it doesn't exist
    eval_script = Path("scripts/evaluate_model.py")
    if not eval_script.exists():
        logger.warning("Evaluation script not found, skipping evaluation")
        return True
    
    cmd = [
        "python", "scripts/evaluate_model.py",
        "--model_path", args.output_dir + "/trained_model",
        "--test_data", args.output_dir + "/training_data/combined_training_data.json",
        "--output_dir", args.output_dir + "/evaluation"
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Model evaluation completed successfully")
        logger.debug(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Model evaluation failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def create_training_summary(config: dict, args, success: bool) -> None:
    """Create training summary report"""
    summary = {
        "training_completed": success,
        "timestamp": datetime.now().isoformat(),
        "config_file": args.config,
        "data_directory": args.data_dir,
        "output_directory": args.output_dir,
        "symbols": config['crypto']['symbols'],
        "model_name": config['model']['name'],
        "max_samples": config['data']['max_samples'],
        "training_epochs": config['training']['epochs'],
        "learning_rate": config['training']['learning_rate'],
        "batch_size": config['training']['batch_size'],
        "lora_rank": config['lora']['r'],
        "lora_alpha": config['lora']['alpha']
    }
    
    summary_file = Path(args.output_dir) / "training_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Training summary saved to: {summary_file}")

def print_next_steps(args, success: bool):
    """Print next steps for the user"""
    logger.info("=" * 60)
    logger.info("NEXT STEPS")
    logger.info("=" * 60)
    
    if success:
        logger.info("✅ Training completed successfully!")
        logger.info("")
        logger.info("Your trained model is ready to use:")
        logger.info(f"  📁 Model location: {args.output_dir}/trained_model")
        logger.info(f"  📊 Training data: {args.output_dir}/training_data")
        logger.info("")
        logger.info("To use your trained model:")
        logger.info("  1. Update config.yaml with your model path:")
        logger.info(f"     model_path: {args.output_dir}/trained_model")
        logger.info("")
        logger.info("  2. Test the model:")
        logger.info("     python -c \"from scripts.test_system import test_system; import asyncio; asyncio.run(test_system())\"")
        logger.info("")
        logger.info("  3. Start trading:")
        logger.info("     python main.py --mode paper --symbols BTC/USD")
        logger.info("")
        logger.info("  4. Monitor performance:")
        logger.info("     python scripts/monitor_dashboard.py")
    else:
        logger.error("❌ Training failed!")
        logger.info("")
        logger.info("Troubleshooting steps:")
        logger.info("  1. Check the logs in logs/ directory")
        logger.info("  2. Verify your data files are complete")
        logger.info("  3. Ensure you have enough GPU memory")
        logger.info("  4. Try reducing batch_size in training_config.yaml")
        logger.info("  5. Check CUDA installation if using GPU")

def main():
    parser = argparse.ArgumentParser(description="Complete Crypto Trading Model Training Pipeline")
    parser.add_argument("--config", type=str, default="training_config.yaml", help="Training configuration file")
    parser.add_argument("--data_dir", type=str, default="./data_1m", help="Directory containing crypto data")
    parser.add_argument("--output_dir", type=str, default="./trained_models", help="Output directory for trained model")
    parser.add_argument("--skip_data_prep", action="store_true", help="Skip data preparation step")
    parser.add_argument("--skip_training", action="store_true", help="Skip model training step")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip model evaluation step")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be done without executing")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("🚀 Starting Crypto Trading Model Training Pipeline")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Check requirements
    if not check_requirements():
        logger.error("Missing required packages. Please install them first.")
        return 1
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Check data availability
    if not check_data_availability(args.data_dir, config['crypto']['symbols']):
        logger.error("Required data files not found")
        return 1
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Dry run mode
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual training will be performed")
        logger.info("Steps that would be executed:")
        if not args.skip_data_prep:
            logger.info("  1. Data preparation")
        if not args.skip_training:
            logger.info("  2. Model training")
        if not args.skip_evaluation:
            logger.info("  3. Model evaluation")
        return 0
    
    # Execute training pipeline
    success = True
    
    # Step 1: Data Preparation
    if not args.skip_data_prep:
        if not run_data_preparation(config, args):
            success = False
            logger.error("Data preparation failed, stopping pipeline")
            create_training_summary(config, args, success)
            print_next_steps(args, success)
            return 1
    
    # Step 2: Model Training
    if not args.skip_training and success:
        if not run_model_training(config, args):
            success = False
            logger.error("Model training failed")
    
    # Step 3: Model Evaluation
    if not args.skip_evaluation and success:
        if not run_model_evaluation(config, args):
            logger.warning("Model evaluation failed, but training completed")
    
    # Create summary and next steps
    create_training_summary(config, args, success)
    print_next_steps(args, success)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
