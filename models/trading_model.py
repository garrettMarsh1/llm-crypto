"""
Trading LLM Model with Quantization and LoRA Support
"""

import torch
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import numpy as np

from loguru import logger


class TradingModel:
    """Trading LLM with quantization and LoRA fine-tuning support"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_model(self, model_path: Optional[str] = None):
        """Load the base model with quantization"""
        try:
            model_name = model_path or self.config['model']['name']
            logger.info(f"Loading model: {model_name}")
            
            # Configure quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.bfloat16,
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            logger.info("✅ Model loaded successfully with 4-bit quantization")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def setup_lora(self, lora_config: Optional[Dict] = None):
        """Setup LoRA for fine-tuning"""
        try:
            if lora_config is None:
                lora_config = self.config['lora']
            
            peft_config = LoraConfig(
                r=lora_config['r'],
                lora_alpha=lora_config['lora_alpha'],
                target_modules=lora_config['target_modules'],
                lora_dropout=lora_config['lora_dropout'],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            self.peft_model = get_peft_model(self.model, peft_config)
            self.peft_model.print_trainable_parameters()
            
            logger.info("✅ LoRA configuration applied successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup LoRA: {e}")
            raise
    
    def load_lora_weights(self, lora_path: str):
        """Load pre-trained LoRA weights"""
        try:
            self.peft_model = PeftModel.from_pretrained(self.model, lora_path)
            logger.info(f"✅ LoRA weights loaded from {lora_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load LoRA weights: {e}")
            raise
    
    def save_lora_weights(self, output_path: str):
        """Save LoRA weights"""
        try:
            self.peft_model.save_pretrained(output_path)
            logger.info(f"✅ LoRA weights saved to {output_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save LoRA weights: {e}")
            raise
    
    def prepare_training_data(self, data: List[Dict[str, str]]) -> Dataset:
        """Prepare training data for fine-tuning"""
        try:
            # Format data for instruction tuning
            formatted_data = []
            for item in data:
                prompt = self._format_trading_prompt(item)
                formatted_data.append({
                    "text": prompt,
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "output": item.get("output", "")
                })
            
            # Tokenize data
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                )
            
            dataset = Dataset.from_list(formatted_data)
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare training data: {e}")
            raise
    
    def _format_trading_prompt(self, item: Dict[str, str]) -> str:
        """Format trading data into instruction prompt"""
        instruction = item.get("instruction", "")
        input_data = item.get("input", "")
        output = item.get("output", "")
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert cryptocurrency trading agent. Analyze market data and provide trading signals with clear reasoning.

<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}

Market Data:
{input_data}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""
        
        return prompt
    
    def fine_tune(self, train_dataset: Dataset, output_dir: str = "./trading_model"):
        """Fine-tune the model with LoRA"""
        try:
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                num_train_epochs=3,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=10,
                save_steps=500,
                evaluation_strategy="no",
                save_total_limit=2,
                remove_unused_columns=False,
                push_to_hub=False,
                report_to="none",
                warmup_steps=100,
                max_steps=1000,
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )
            
            # Trainer
            trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
            )
            
            # Train
            logger.info("Starting fine-tuning...")
            trainer.train()
            
            # Save model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"✅ Fine-tuning completed. Model saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Fine-tuning failed: {e}")
            raise
    
    def generate_signal(self, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal based on market context"""
        try:
            # Build prompt from market context
            prompt = self._build_trading_prompt(market_context)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.peft_model.generate(
                    **inputs,
                    max_new_tokens=self.config['model']['max_tokens'],
                    temperature=self.config['model']['temperature'],
                    top_p=self.config['model']['top_p'],
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse signal from response
            signal = self._parse_trading_signal(response, market_context)
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "reasoning": f"Error generating signal: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _build_trading_prompt(self, market_context: Dict[str, Any]) -> str:
        """Build trading prompt from market context"""
        symbol = market_context.get('symbol', 'BTC/USD')
        bars = market_context.get('bars', [])
        quotes = market_context.get('quotes', [])
        positions = market_context.get('positions', [])
        
        # Format recent bars
        bars_text = ""
        if bars:
            bars_text = "Recent Price Bars:\n"
            for bar in bars[-5:]:  # Last 5 bars
                bars_text += f"Time: {bar['timestamp']}, Open: {bar['open']}, High: {bar['high']}, Low: {bar['low']}, Close: {bar['close']}, Volume: {bar['volume']}\n"
        
        # Format current positions
        positions_text = ""
        if positions:
            positions_text = "Current Positions:\n"
            for pos in positions:
                positions_text += f"Symbol: {pos['symbol']}, Qty: {pos['qty']}, Side: {pos['side']}, P&L: {pos['unrealized_pl']}\n"
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert cryptocurrency trading agent. Analyze the provided market data and generate a trading signal with clear reasoning.

Trading Rules:
- Only trade BTC/USD, ETH/USD, or SOL/USD
- Maximum position size: 10% of portfolio
- Consider risk management and market volatility
- Provide clear reasoning for your decision

<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze the following market data for {symbol} and provide a trading signal:

{bars_text}

{positions_text}

Current Market Context:
- Symbol: {symbol}
- Analysis Time: {datetime.now().isoformat()}

Please provide:
1. Trading action (buy/sell/hold)
2. Confidence level (0-1)
3. Reasoning for the decision
4. Risk assessment

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return prompt
    
    def _parse_trading_signal(self, response: str, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse trading signal from model response"""
        try:
            # Extract action
            action = "hold"
            if "buy" in response.lower():
                action = "buy"
            elif "sell" in response.lower():
                action = "sell"
            
            # Extract confidence (look for numbers 0-1)
            import re
            confidence_match = re.search(r'confidence[:\s]*([0-9]*\.?[0-9]+)', response.lower())
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            
            # Extract reasoning
            reasoning_start = response.find("reasoning")
            if reasoning_start != -1:
                reasoning = response[reasoning_start:reasoning_start+200]
            else:
                reasoning = response[:200]
            
            return {
                "action": action,
                "confidence": min(max(confidence, 0.0), 1.0),
                "reasoning": reasoning.strip(),
                "symbol": market_context.get('symbol', 'BTC/USD'),
                "timestamp": datetime.now().isoformat(),
                "raw_response": response
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to parse trading signal: {e}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "reasoning": f"Error parsing signal: {str(e)}",
                "symbol": market_context.get('symbol', 'BTC/USD'),
                "timestamp": datetime.now().isoformat(),
                "raw_response": response
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.peft_model is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_name": self.config['model']['name'],
            "quantization": "4-bit",
            "lora_enabled": True,
            "device": str(self.device),
            "trainable_parameters": self.peft_model.get_nb_trainable_parameters(),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "memory_usage": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        }


def create_sample_training_data() -> List[Dict[str, str]]:
    """Create sample training data for crypto trading"""
    return [
        {
            "instruction": "Analyze the following BTC/USD market data and provide a trading signal.",
            "input": "Recent bars show: Open: 45000, High: 46000, Low: 44500, Close: 45500, Volume: 1000. RSI: 65, MACD: Bullish crossover.",
            "output": "Based on the technical analysis, I recommend a BUY signal with 0.7 confidence. The MACD bullish crossover and RSI at 65 indicate upward momentum. However, consider the high volatility and set appropriate stop-loss at 44500."
        },
        {
            "instruction": "Evaluate ETH/USD market conditions and suggest trading action.",
            "input": "Price: 3200, Volume: Decreasing, Support: 3100, Resistance: 3300, Market sentiment: Bearish due to regulatory concerns.",
            "output": "I recommend HOLD with 0.6 confidence. The decreasing volume and bearish sentiment suggest caution. Wait for a clear break above resistance at 3300 or below support at 3100 before taking action."
        },
        {
            "instruction": "Assess SOL/USD trading opportunity based on current market data.",
            "input": "Price: 95, 24h change: -5%, Volume: High, Technical indicators: Oversold conditions, Support: 90.",
            "output": "BUY signal with 0.8 confidence. The oversold conditions and strong support at 90 present a good entry opportunity. The high volume suggests institutional interest. Set stop-loss at 88 and target 105."
        }
    ]


if __name__ == "__main__":
    # Test the model
    model = TradingModel()
    model.load_model()
    model.setup_lora()
    
    # Test signal generation
    test_context = {
        "symbol": "BTC/USD",
        "bars": [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "open": 45000,
                "high": 46000,
                "low": 44500,
                "close": 45500,
                "volume": 1000
            }
        ],
        "positions": []
    }
    
    signal = model.generate_signal(test_context)
    print("Generated Signal:", signal)
