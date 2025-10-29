# 🚀 Crypto Trading Model Training Guide

This guide will walk you through fine-tuning a Large Language Model (LLM) on cryptocurrency trading data to create an intelligent trading agent.

## 📋 Overview

We'll fine-tune **Qwen2.5-7B-Instruct** on your crypto data using:
- **LoRA (Low-Rank Adaptation)** for efficient fine-tuning
- **4-bit quantization** to fit on your RTX 4070 (12GB VRAM)
- **Advanced technical indicators** for comprehensive market analysis
- **Multi-symbol training** on BTC, ETH, SOL, DOGE data

## 🎯 What You'll Get

After training, you'll have a specialized model that can:
- Analyze crypto market data in real-time
- Generate trading signals (BUY/SELL/HOLD)
- Provide detailed reasoning for each decision
- Suggest position sizing and risk management
- Adapt to different market conditions

## 📊 Training Data

Your model will be trained on:
- **5 years** of 1-minute OHLCV data
- **Multiple cryptocurrencies**: BTC, ETH, SOL, DOGE
- **Technical indicators**: RSI, MACD, Bollinger Bands, Stochastic, etc.
- **Market context**: Volume, volatility, trends
- **Trading signals**: Generated using advanced technical analysis

## 🛠️ Prerequisites

### Hardware Requirements
- **GPU**: RTX 4070 (12GB VRAM) or better
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space
- **CPU**: Multi-core processor (i7-10700K is perfect)

### Software Requirements
- Python 3.10+
- CUDA 12.0+
- PyTorch with CUDA support
- All dependencies from `requirements.txt`

## 🚀 Quick Start

### 1. Download Crypto Data
```bash
# Download 5 years of 1-minute data for BTC, ETH, SOL, DOGE
python data_for_training.py
```

### 2. Start Training
```bash
# Run the complete training pipeline
python start_training.py
```

### 3. Monitor Progress
```bash
# Check training logs
tail -f logs/training_pipeline_*.log

# Monitor GPU usage
nvidia-smi
```

## 📁 Training Pipeline

The training process consists of 3 main steps:

### Step 1: Data Preparation (`scripts/prepare_training_data.py`)
- Loads your crypto CSV files
- Calculates 20+ technical indicators
- Generates training samples every 4 hours
- Creates signal labels using advanced analysis
- Saves processed data in JSON format

**Output**: `./training_data/combined_training_data.json`

### Step 2: Model Training (`scripts/train_crypto_model.py`)
- Loads Qwen2.5-7B with 4-bit quantization
- Applies LoRA for efficient fine-tuning
- Trains on crypto trading prompts
- Saves checkpoints every 500 steps
- Evaluates on test data

**Output**: `./trained_models/trained_model/`

### Step 3: Model Evaluation (`scripts/evaluate_model.py`)
- Tests model on unseen data
- Calculates trading performance metrics
- Generates evaluation reports
- Creates performance visualizations

**Output**: `./trained_models/evaluation/`

## ⚙️ Configuration

### Training Configuration (`training_config.yaml`)

```yaml
# Model settings
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  max_length: 1024

# LoRA settings (adjust for your GPU)
lora:
  r: 16                    # Higher = more parameters
  alpha: 32                # Typically 2x the rank
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

# Training settings
training:
  batch_size: 2            # Adjust based on GPU memory
  learning_rate: 2e-4      # Learning rate for LoRA
  epochs: 3                # Number of training epochs
  gradient_accumulation_steps: 8  # Effective batch size
```

### Memory Optimization

For RTX 4070 (12GB VRAM):
```yaml
training:
  batch_size: 2            # Start with 2, increase if memory allows
  gradient_accumulation_steps: 8  # Effective batch size = 16
  fp16: true               # Use 16-bit precision
  gradient_checkpointing: true  # Save memory
```

## 📈 Training Process

### Data Flow
1. **Raw Data**: 1-minute OHLCV from Coinbase
2. **Technical Analysis**: 20+ indicators calculated
3. **Signal Generation**: BUY/SELL/HOLD based on analysis
4. **Prompt Creation**: Structured prompts for LLM training
5. **Model Training**: LoRA fine-tuning on trading data

### Training Samples
Each training sample includes:
```json
{
  "symbol": "BTC-USD",
  "timestamp": "2024-01-01T10:00:00Z",
  "price_data": {
    "open": 45000.0,
    "high": 46000.0,
    "low": 44500.0,
    "close": 45500.0,
    "volume": 1000.0,
    "change_24h": 2.5
  },
  "technical_indicators": {
    "rsi": 65.2,
    "macd": 150.5,
    "bb_position": 0.7,
    "stoch_k": 75.8
  },
  "signal": "BUY",
  "confidence": 0.85,
  "reasoning": "Strong bullish momentum with RSI in neutral zone..."
}
```

## 🎛️ Advanced Configuration

### Custom Training Parameters

```bash
# Train with custom settings
python scripts/run_training_pipeline.py \
  --config custom_config.yaml \
  --data_dir ./my_data \
  --output_dir ./my_models \
  --max_samples 50000
```

### Memory Optimization Tips

1. **Reduce Batch Size**: If you get OOM errors
2. **Increase Gradient Accumulation**: Maintain effective batch size
3. **Use Gradient Checkpointing**: Trade speed for memory
4. **Enable FP16**: Use half precision
5. **Reduce Sequence Length**: Shorter prompts use less memory

### Multi-GPU Training

```yaml
# For multiple GPUs
training:
  dataloader_num_workers: 4
  ddp_find_unused_parameters: false
  ddp_backend: "nccl"
```

## 📊 Monitoring Training

### Real-time Monitoring
```bash
# Watch training logs
tail -f logs/training_*.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check training progress
tensorboard --logdir logs/tensorboard
```

### Key Metrics to Watch
- **Loss**: Should decrease over time
- **Learning Rate**: Follows cosine schedule
- **GPU Memory**: Should stay under 12GB
- **Training Speed**: Samples per second
- **Validation Loss**: Should track training loss

## 🔧 Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)
```yaml
# Reduce memory usage
training:
  batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  fp16: true
```

#### 2. Slow Training
```yaml
# Optimize for speed
training:
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  remove_unused_columns: false
```

#### 3. Poor Performance
- Increase training epochs
- Adjust learning rate
- Add more training data
- Tune LoRA parameters

### Debug Mode
```bash
# Run with debug logging
python scripts/run_training_pipeline.py --config training_config.yaml --dry_run
```

## 📈 Expected Results

### Training Time
- **Data Preparation**: 5-15 minutes
- **Model Training**: 2-6 hours (RTX 4070)
- **Evaluation**: 10-30 minutes
- **Total**: 3-7 hours

### Model Performance
After training, your model should:
- Generate coherent trading signals
- Provide detailed reasoning
- Adapt to market conditions
- Maintain consistent performance

### Sample Output
```
**Trading Signal: BUY**

**Reasoning:**
Based on my analysis of the BTC-USD market data:
- RSI indicates oversold conditions at 28.5
- MACD shows bullish momentum above zero line
- Price near lower Bollinger Band (oversold)
- High volume confirms buying pressure
- Strong 24h uptrend of 3.2%

**Key Factors:**
- Price Action: $45,500 (+2.1%)
- Technical Analysis: 28.5 RSI, 150.5 MACD
- Market Sentiment: Fear (Bearish trend)

**Risk Assessment:**
- Position Size: 2-3% of portfolio
- Stop Loss: 2.5% below entry
- Take Profit: 6-8% above entry
```

## 🎯 Next Steps

After training completes:

1. **Test Your Model**:
   ```bash
   python -c "from scripts.test_system import test_system; import asyncio; asyncio.run(test_system())"
   ```

2. **Update Configuration**:
   ```yaml
   model:
     path: "./trained_models/trained_model"
   ```

3. **Start Trading**:
   ```bash
   python main.py --mode paper --symbols BTC/USD
   ```

4. **Monitor Performance**:
   ```bash
   python scripts/monitor_dashboard.py
   ```

## 📚 Additional Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Qwen2.5 Documentation](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Technical Analysis Guide](https://www.investopedia.com/technical-analysis-4689657)

## 🆘 Support

If you encounter issues:
1. Check the logs in `logs/` directory
2. Verify your data files are complete
3. Ensure sufficient GPU memory
4. Review the troubleshooting section above

Happy training! 🚀
