#!/usr/bin/env python3
"""
Run Fast JSON to Parquet Conversion
Kills slow process and runs fast ijson version
"""

import subprocess
import sys
import time
from pathlib import Path
from loguru import logger
import psutil

def kill_slow_processes():
    """Kill any running parquet processes"""
    logger.info("Looking for slow parquet processes...")
    
    killed_count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'parquet_chunk_processor.py' in cmdline:
                logger.info(f"Killing slow process: PID {proc.info['pid']}")
                proc.kill()
                killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if killed_count > 0:
        logger.info(f"Killed {killed_count} slow processes")
        time.sleep(2)  # Wait for processes to die
    else:
        logger.info("No slow processes found")

def install_ijson():
    """Install ijson if not available"""
    try:
        import ijson
        logger.info("ijson already installed")
        return True
    except ImportError:
        logger.info("Installing ijson...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "ijson"], check=True)
            logger.info("ijson installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install ijson: {e}")
            return False

def main():
    """Main function"""
    
    logger.info("Starting FAST Parquet conversion...")
    
    # Kill slow processes
    kill_slow_processes()
    
    # Install ijson
    if not install_ijson():
        logger.error("Cannot install ijson. Exiting.")
        return
    
    # Check input file
    input_file = Path("training_data/BTC_USD_training_data.json")
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    file_size_gb = input_file.stat().st_size / (1024**3)
    logger.info(f"Input file size: {file_size_gb:.2f}GB")
    
    # Run fast conversion
    logger.info("Running FAST conversion with ijson...")
    
    cmd = [
        "crypto_trading_env\\Scripts\\python.exe", "scripts/fast_parquet_processor.py",
        "--input_file", str(input_file),
        "--output_dir", "./parquet_chunks",
        "--chunk_size", "10000",
        "--max_memory", "0.8"
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"FAST conversion completed in {duration:.1f} seconds!")
        logger.info(f"Output: {result.stdout}")
        
        # Check results
        chunks_dir = Path("./parquet_chunks")
        if chunks_dir.exists():
            chunk_files = list(chunks_dir.glob("chunk_*.parquet"))
            logger.info(f"Created {len(chunk_files)} Parquet chunks")
            
            # Calculate compression
            input_size = input_file.stat().st_size
            output_size = sum(f.stat().st_size for f in chunk_files)
            compression_ratio = (1 - output_size / input_size) * 100
            
            logger.info(f"Compression ratio: {compression_ratio:.1f}%")
            logger.info(f"Output size: {output_size / (1024**3):.2f}GB")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Fast conversion failed: {e}")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")

if __name__ == "__main__":
    main()
