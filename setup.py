#!/usr/bin/env python3
"""
Setup script for Crypto Trading LLM package
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements_parquet.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="crypto-trading-llm",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Memory-efficient LLM fine-tuning for cryptocurrency trading analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/crypto-trading-llm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "colab": [
            "google-colab>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crypto-train=scripts.parquet_chunk_trainer:main",
            "crypto-convert=scripts.fast_parquet_processor:main",
            "crypto-pipeline=run_parquet_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    keywords="cryptocurrency, trading, llm, fine-tuning, parquet, memory-efficient",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/crypto-trading-llm/issues",
        "Source": "https://github.com/yourusername/crypto-trading-llm",
        "Documentation": "https://github.com/yourusername/crypto-trading-llm#readme",
    },
)