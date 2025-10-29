"""
Crypto Trading LLM Package
Memory-efficient LLM fine-tuning for cryptocurrency trading analysis
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .scripts.parquet_chunk_trainer import ParquetChunkTrainer
from .scripts.fast_parquet_processor import FastParquetProcessor

__all__ = [
    "ParquetChunkTrainer",
    "FastParquetProcessor",
]
