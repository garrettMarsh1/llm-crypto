# Crypto Trading LLM 🚀

A memory-efficient LLM fine-tuning package for cryptocurrency trading analysis. Train large language models on massive datasets without running out of RAM!

## Features

- **Memory Efficient**: Process 7GB+ datasets with only 8-16GB RAM
- **Parquet Chunks**: Convert large JSON files to optimized Parquet chunks
- **4-bit Quantization**: Use QLoRA for efficient training
- **Google Colab Ready**: Easy installation and usage in Colab
- **Production Ready**: No mocks, no simplifications - real implementation

## Quick Start

### Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/crypto-trading-llm/blob/main/colab_quickstart.ipynb)

1. Open the Colab notebook above
2. Upload your `BTC_USD_training_data.json` file
3. Run all cells
4. Get your trained model in 1-2 hours!

### Local Installation

```bash
# Install from GitHub
pip install git+https://github.com/yourusername/crypto-trading-llm.git

# Or clone and install locally
git clone https://github.com/yourusername/crypto-trading-llm.git
cd crypto-trading-llm
pip install -e .
```

## Usage

### 1. Convert JSON to Parquet Chunks

```python
from crypto_trading_llm import FastParquetProcessor

processor = FastParquetProcessor(chunk_size=10000, max_memory_usage=0.8)
results = processor.process_json_with_ijson(
    input_file="BTC_USD_training_data.json",
    output_dir="./parquet_chunks"
)
```

### 2. Train the Model

```python
from crypto_trading_llm import ParquetChunkTrainer

trainer = ParquetChunkTrainer(max_memory_usage=0.85)
trainer.train_on_chunks(
    data_dir="./parquet_chunks",
    output_dir="./trained_model",
    epochs_per_chunk=1,
    max_chunks=5  # Start with 5 chunks for testing
)
```

### 3. Use Command Line Tools

```bash
# Convert JSON to Parquet
crypto-convert --input_file BTC_USD_training_data.json --output_dir ./parquet_chunks

# Train model
crypto-train --data_dir ./parquet_chunks --output_dir ./trained_model --max_chunks 5

# Run full pipeline
crypto-pipeline --input_file BTC_USD_training_data.json --output_dir ./trained_model
```

## Performance

### Memory Usage
- **7.11GB JSON file**: Uses only 8-12GB RAM
- **2.6M samples**: Processes in 10K chunks
- **Compression**: 88.7% size reduction (7.11GB → 0.80GB)

### Training Speed
- **Per chunk**: ~2-3 minutes
- **5 chunks (test)**: ~10-15 minutes
- **All 263 chunks**: ~8-13 hours
- **GPU recommended**: T4, V100, or A100

## Architecture

```
Input JSON (7.11GB)
    ↓
Fast ijson Parser
    ↓
Parquet Chunks (0.80GB)
    ↓
Chunked Training
    ↓
Trained Model
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- 8GB+ RAM (16GB+ recommended)
- GPU recommended for training

## Installation

### Dependencies

```bash
pip install pyarrow pandas numpy transformers torch peft datasets bitsandbytes accelerate loguru psutil ijson
```

### Development

```bash
git clone https://github.com/yourusername/crypto-trading-llm.git
cd crypto-trading-llm
pip install -e ".[dev]"
```

## Data Format

Your JSON file should contain an array of trading samples:

```json
[
  {
    "symbol": "BTC-USD",
    "price_data": {
      "open": 50000.0,
      "high": 51000.0,
      "low": 49000.0,
      "close": 50500.0,
      "volume": 1000000,
      "change_24h": 1.0
    },
    "technical_indicators": {
      "rsi": 65.2,
      "macd": 0.15,
      "macd_signal": 0.12,
      "bb_position": 0.7,
      "volume_ratio": 1.2
    },
    "market_context": {
      "market_cap": 850000000000,
      "fear_greed": "Greed",
      "trend": "Bullish"
    },
    "signal": "BUY",
    "confidence": 0.85,
    "reasoning": "Strong technical indicators with bullish momentum"
  }
]
```

## Model Output

The trained model will generate responses in this format:

```
SIGNAL: BUY
CONFIDENCE: 0.85
REASONING: Strong technical indicators with bullish momentum. RSI at 65.2 shows healthy momentum, MACD bullish crossover, and price above key support levels.
```

## Advanced Usage

### Custom Model

```python
trainer = ParquetChunkTrainer(
    model_name="microsoft/DialoGPT-medium",  # Use different base model
    max_memory_usage=0.8
)
```

### Custom Chunk Size

```python
processor = FastParquetProcessor(
    chunk_size=20000,  # Larger chunks for more memory
    max_memory_usage=0.9
)
```

### News Sentiment Integration

```python
# Add news sentiment to your data
news_data = {
    "headline": "Bitcoin ETF Approval Expected",
    "sentiment_score": 0.8,
    "impact_score": 0.9
}
```

## Troubleshooting

### Memory Issues
- Reduce `chunk_size` to 5000
- Lower `max_memory_usage` to 0.7
- Use smaller model (e.g., DialoGPT-medium)

### Training Issues
- Check GPU availability: `torch.cuda.is_available()`
- Monitor memory usage during training
- Use gradient checkpointing (enabled by default)

### Colab Issues
- Restart runtime if memory is full
- Use Pro/Pro+ for better GPUs
- Save checkpoints frequently

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/yourusername/crypto-trading-llm/issues)
- Documentation: [Full documentation](https://github.com/yourusername/crypto-trading-llm#readme)

## Roadmap

- [ ] News sentiment integration
- [ ] Real-time data processing
- [ ] Model deployment tools
- [ ] Performance benchmarks
- [ ] Multi-asset support

---

**Made with ❤️ for the crypto trading community**