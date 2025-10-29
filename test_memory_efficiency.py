#!/usr/bin/env python3
"""
Test Memory Efficiency
Test the memory-efficient processing with small datasets first
"""

import psutil
import time
from pathlib import Path
from loguru import logger
import subprocess
import sys

def get_memory_usage():
    """Get current memory usage"""
    return psutil.virtual_memory().percent

def test_memory_efficiency():
    """Test memory efficiency with small dataset"""
    
    logger.info("Testing memory efficiency...")
    
    # Check initial memory
    initial_memory = get_memory_usage()
    logger.info(f"Initial memory usage: {initial_memory:.1f}%")
    
    # Test with very small chunks first
    logger.info("Testing with minimal settings...")
    
    cmd = [
        "python", "scripts/memory_efficient_data_processor.py",
        "--data_dir", "./data_1m",
        "--output_dir", "./test_training_data",
        "--symbols", "BTC-USD",  # Only one symbol for testing
        "--chunk_size", "1000",  # Very small chunks
        "--sample_interval", "1440",  # 24 hours (fewer samples)
        "--max_samples", "100",  # Very few samples
        "--max_memory", "0.7"  # Conservative memory threshold
    ]
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        logger.info(f"Processing completed in {end_time - start_time:.1f} seconds")
        logger.info(f"Memory usage: {start_memory:.1f}% -> {end_memory:.1f}%")
        logger.info(f"Peak memory during processing: {max(start_memory, end_memory):.1f}%")
        
        if result.stdout:
            logger.info("Output:")
            logger.info(result.stdout)
        
        # Check if output files were created
        output_dir = Path("./test_training_data")
        if output_dir.exists():
            files = list(output_dir.glob("*.json"))
            logger.info(f"Created {len(files)} output files:")
            for file in files:
                logger.info(f"  - {file.name}")
        
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("Processing timed out after 5 minutes")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Processing failed: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

def main():
    """Main test function"""
    
    logger.info("Starting memory efficiency test...")
    
    # Check if data directory exists
    data_dir = Path("./data_1m")
    if not data_dir.exists():
        logger.error("Data directory not found. Please ensure data_1m directory exists with CSV files.")
        return
    
    # Check available memory
    memory = psutil.virtual_memory()
    logger.info(f"Available memory: {memory.total / (1024**3):.1f} GB")
    logger.info(f"Available memory: {memory.available / (1024**3):.1f} GB")
    
    if memory.available < 2 * (1024**3):  # Less than 2GB available
        logger.warning("Low available memory detected. Consider closing other applications.")
    
    # Run test
    success = test_memory_efficiency()
    
    if success:
        logger.info("Memory efficiency test PASSED!")
        logger.info("You can now run the full processing with confidence.")
    else:
        logger.error("Memory efficiency test FAILED!")
        logger.error("Please check the error messages above and adjust settings.")

if __name__ == "__main__":
    main()

