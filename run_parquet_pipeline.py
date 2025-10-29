#!/usr/bin/env python3
"""
Parquet Pipeline Runner
Complete pipeline: JSON -> Parquet chunks -> Training
"""

import subprocess
import sys
import time
from pathlib import Path
from loguru import logger
import psutil

def get_memory_info():
    """Get current memory information"""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_percent': memory.percent
    }

def run_command(cmd, description, timeout=None):
    """Run a command and handle errors"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Success: {description} (took {duration:.1f}s)")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        
        return True, duration
        
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout: {description} (exceeded {timeout}s)")
        return False, None
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Error: {e.stderr}")
        return False, None
    except Exception as e:
        logger.error(f"Unexpected error in {description}: {e}")
        return False, None

def check_dependencies():
    """Check if required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'pyarrow',
        'transformers', 
        'peft',
        'datasets',
        'bitsandbytes',
        'accelerate'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Install with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("All dependencies found")
    return True

def main():
    """Main pipeline function"""
    
    logger.info("Starting Parquet Pipeline...")
    
    # Check memory
    memory_info = get_memory_info()
    logger.info(f"System memory: {memory_info['total_gb']:.1f}GB total, "
               f"{memory_info['available_gb']:.1f}GB available, "
               f"{memory_info['used_percent']:.1f}% used")
    
    if memory_info['available_gb'] < 4:
        logger.warning("Low available memory. Consider closing other applications.")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing dependencies. Please install required packages.")
        return
    
    # Check input file
    input_file = Path("training_data/BTC_USD_training_data.json")
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.info("Please ensure BTC_USD_training_data.json exists in training_data/")
        return
    
    # Check file size
    file_size_gb = input_file.stat().st_size / (1024**3)
    logger.info(f"Input file size: {file_size_gb:.2f}GB")
    
    if file_size_gb > 10:
        logger.warning("Large file detected. This may take a while...")
    
    # Step 1: Convert JSON to Parquet chunks
    logger.info("="*60)
    logger.info("STEP 1: Converting JSON to Parquet chunks")
    logger.info("="*60)
    
    cmd1 = [
        "python", "scripts/parquet_chunk_processor.py",
        "--input_file", str(input_file),
        "--output_dir", "./parquet_chunks",
        "--chunk_size", "10000",  # 10K samples per chunk
        "--max_memory", "0.8"
    ]
    
    success1, duration1 = run_command(cmd1, "JSON to Parquet conversion", timeout=3600)  # 1 hour timeout
    
    if not success1:
        logger.error("Step 1 failed! Cannot proceed.")
        return
    
    # Check if chunks were created
    chunks_dir = Path("./parquet_chunks")
    if not chunks_dir.exists():
        logger.error("Parquet chunks directory not created!")
        return
    
    chunk_files = list(chunks_dir.glob("chunk_*.parquet"))
    if not chunk_files:
        logger.error("No Parquet chunks created!")
        return
    
    logger.info(f"Created {len(chunk_files)} Parquet chunks")
    
    # Calculate compression ratio
    total_chunk_size = sum(f.stat().st_size for f in chunk_files)
    compression_ratio = (1 - total_chunk_size / input_file.stat().st_size) * 100
    logger.info(f"Compression ratio: {compression_ratio:.1f}%")
    logger.info(f"Total chunk size: {total_chunk_size / (1024**3):.2f}GB")
    
    # Step 2: Train model on Parquet chunks
    logger.info("="*60)
    logger.info("STEP 2: Training model on Parquet chunks")
    logger.info("="*60)
    
    cmd2 = [
        "python", "scripts/parquet_chunk_trainer.py",
        "--data_dir", "./parquet_chunks",
        "--output_dir", "./parquet_trained_model",
        "--epochs_per_chunk", "1",
        "--max_memory", "0.8"
    ]
    
    # For testing, limit to first 5 chunks
    if len(chunk_files) > 5:
        cmd2.extend(["--max_chunks", "5"])
        logger.info("Limiting to first 5 chunks for testing")
    
    success2, duration2 = run_command(cmd2, "Parquet chunk training", timeout=7200)  # 2 hour timeout
    
    if not success2:
        logger.error("Step 2 failed!")
        return
    
    # Summary
    logger.info("="*60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    
    total_duration = (duration1 or 0) + (duration2 or 0)
    logger.info(f"Total processing time: {total_duration:.1f}s")
    logger.info(f"Input file: {file_size_gb:.2f}GB")
    logger.info(f"Output chunks: {len(chunk_files)} files, {total_chunk_size / (1024**3):.2f}GB")
    logger.info(f"Compression: {compression_ratio:.1f}%")
    
    logger.info("Results:")
    logger.info("  - Parquet chunks: ./parquet_chunks/")
    logger.info("  - Trained model: ./parquet_trained_model/")
    logger.info("  - Metadata: ./parquet_chunks/metadata.json")
    
    # Show chunk information
    logger.info("Chunk files:")
    for i, chunk_file in enumerate(chunk_files[:10]):  # Show first 10
        size_mb = chunk_file.stat().st_size / (1024**2)
        logger.info(f"  {chunk_file.name}: {size_mb:.1f}MB")
    
    if len(chunk_files) > 10:
        logger.info(f"  ... and {len(chunk_files) - 10} more chunks")

if __name__ == "__main__":
    main()
