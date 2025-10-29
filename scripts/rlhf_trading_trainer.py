#!/usr/bin/env python3
"""
RLHF Training System for Crypto Trading Model
Implements process supervision and reward modeling for trading decisions
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
from loguru import logger
import pandas as pd
from datetime import datetime

@dataclass
class TradingStep:
    """Represents a single step in trading analysis"""
    step_type: str  # 'technical_analysis', 'risk_assessment', 'market_context', 'signal_generation'
    content: str
    confidence: float
    reasoning: str
    is_correct: bool = None
    reward: float = 0.0

@dataclass
class TradingDecision:
    """Complete trading decision with all steps"""
    market_data: Dict
    steps: List[TradingStep]
    final_signal: str
    final_confidence: float
    final_reasoning: str
    actual_outcome: float = None  # P&L after 24h
    total_reward: float = 0.0

class TradingRewardModel(nn.Module):
    """Reward model for evaluating trading analysis steps"""
    
    def __init__(self, model_name: str, hidden_size: int = 768):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Freeze the base model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Use the last hidden state of the last token
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [batch_size, hidden_size]
        reward = self.reward_head(last_hidden)
        return reward

class TradingStepDataset(Dataset):
    """Dataset for training the reward model"""
    
    def __init__(self, decisions: List[TradingDecision], tokenizer, max_length: int = 512):
        self.decisions = decisions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._create_samples()
    
    def _create_samples(self):
        samples = []
        for decision in self.decisions:
            for step in decision.steps:
                # Create prompt for this step
                prompt = self._create_step_prompt(decision.market_data, step)
                
                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                # Calculate reward based on step correctness and outcome
                reward = self._calculate_step_reward(step, decision)
                
                samples.append({
                    'input_ids': inputs['input_ids'].squeeze(),
                    'attention_mask': inputs['attention_mask'].squeeze(),
                    'reward': reward,
                    'step_type': step.step_type,
                    'is_correct': step.is_correct
                })
        
        return samples
    
    def _create_step_prompt(self, market_data: Dict, step: TradingStep) -> str:
        """Create prompt for a specific analysis step"""
        
        base_prompt = f"""<|im_start|>system
You are an expert cryptocurrency trading analyst. Analyze the given market data step by step.

<|im_start|>user
Analyze this {market_data['symbol']} trading data:

Price: ${market_data['close']:.2f} ({market_data['change_24h']:+.2f}%)
Volume: {market_data['volume']:,.0f}
RSI: {market_data['rsi']:.1f}
MACD: {market_data['macd']:.2f}

Current step: {step.step_type}
<|im_end|>

<|im_start|>assistant
{step.content}

Reasoning: {step.reasoning}
Confidence: {step.confidence:.2f}
<|im_end|>"""
        
        return base_prompt
    
    def _calculate_step_reward(self, step: TradingStep, decision: TradingDecision) -> float:
        """Calculate reward for a trading step"""
        base_reward = 0.0
        
        # Base reward for correctness
        if step.is_correct:
            base_reward += 0.5
        
        # Reward for confidence calibration
        if step.is_correct and step.confidence > 0.7:
            base_reward += 0.2  # Bonus for high confidence when correct
        elif not step.is_correct and step.confidence < 0.3:
            base_reward += 0.1  # Bonus for low confidence when wrong
        
        # Reward based on actual trading outcome
        if decision.actual_outcome is not None:
            if decision.actual_outcome > 0:  # Profitable trade
                if step.step_type == 'signal_generation' and step.content.upper() in ['BUY', 'SELL']:
                    base_reward += 0.3
            else:  # Losing trade
                if step.step_type == 'signal_generation' and step.content.upper() == 'HOLD':
                    base_reward += 0.2
        
        # Step-specific rewards
        if step.step_type == 'technical_analysis':
            # Reward for mentioning relevant indicators
            if any(indicator in step.content.lower() for indicator in ['rsi', 'macd', 'bollinger', 'support', 'resistance']):
                base_reward += 0.1
        
        elif step.step_type == 'risk_assessment':
            # Reward for mentioning risk factors
            if any(risk in step.content.lower() for risk in ['volatility', 'stop loss', 'position size', 'risk']):
                base_reward += 0.1
        
        return min(1.0, max(0.0, base_reward))  # Clamp between 0 and 1
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class RLHFTradingTrainer:
    """Main RLHF trainer for crypto trading model"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['name'])
        
        # Load the base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name'],
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load LoRA weights if available
        if os.path.exists(self.config['rlhf']['lora_path']):
            self.model = PeftModel.from_pretrained(self.model, self.config['rlhf']['lora_path'])
        
        # Initialize reward model
        self.reward_model = TradingRewardModel(
            self.config['model']['name'],
            self.model.config.hidden_size
        ).to(self.device)
        
        # Optimizers
        self.model_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['rlhf']['learning_rate']
        )
        self.reward_optimizer = optim.AdamW(
            self.reward_model.parameters(),
            lr=self.config['rlhf']['reward_learning_rate']
        )
        
        # Loss functions
        self.reward_loss_fn = nn.MSELoss()
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
    
    def generate_trading_analysis(self, market_data: Dict) -> TradingDecision:
        """Generate step-by-step trading analysis"""
        
        steps = []
        
        # Step 1: Technical Analysis
        tech_prompt = self._create_technical_analysis_prompt(market_data)
        tech_response = self._generate_response(tech_prompt)
        tech_step = TradingStep(
            step_type="technical_analysis",
            content=tech_response,
            confidence=0.8,  # Will be refined by reward model
            reasoning="Analyzing technical indicators and price patterns"
        )
        steps.append(tech_step)
        
        # Step 2: Risk Assessment
        risk_prompt = self._create_risk_assessment_prompt(market_data, tech_response)
        risk_response = self._generate_response(risk_prompt)
        risk_step = TradingStep(
            step_type="risk_assessment",
            content=risk_response,
            confidence=0.7,
            reasoning="Evaluating risk factors and position sizing"
        )
        steps.append(risk_step)
        
        # Step 3: Market Context
        context_prompt = self._create_market_context_prompt(market_data, tech_response, risk_response)
        context_response = self._generate_response(context_prompt)
        context_step = TradingStep(
            step_type="market_context",
            content=context_response,
            confidence=0.6,
            reasoning="Analyzing broader market conditions and sentiment"
        )
        steps.append(context_step)
        
        # Step 4: Signal Generation
        signal_prompt = self._create_signal_generation_prompt(market_data, steps)
        signal_response = self._generate_response(signal_prompt)
        signal_step = TradingStep(
            step_type="signal_generation",
            content=signal_response,
            confidence=0.9,
            reasoning="Synthesizing all analysis into final trading decision"
        )
        steps.append(signal_step)
        
        # Extract final signal and reasoning
        final_signal = self._extract_signal(signal_response)
        final_reasoning = self._extract_reasoning(signal_response)
        
        return TradingDecision(
            market_data=market_data,
            steps=steps,
            final_signal=final_signal,
            final_confidence=0.8,
            final_reasoning=final_reasoning
        )
    
    def _create_technical_analysis_prompt(self, market_data: Dict) -> str:
        """Create prompt for technical analysis step"""
        return f"""<|im_start|>system
You are a technical analysis expert. Analyze the given crypto market data and provide detailed technical analysis.

<|im_start|>user
Analyze the technical indicators for {market_data['symbol']}:

Price: ${market_data['close']:.2f} ({market_data['change_24h']:+.2f}%)
Volume: {market_data['volume']:,.0f}
RSI: {market_data['rsi']:.1f}
MACD: {market_data['macd']:.2f}
Bollinger Position: {market_data['bb_position']:.2f}

Provide detailed technical analysis focusing on:
1. Trend analysis
2. Momentum indicators
3. Support and resistance levels
4. Volume analysis
<|im_end|>

<|im_start|>assistant
"""
    
    def _create_risk_assessment_prompt(self, market_data: Dict, tech_analysis: str) -> str:
        """Create prompt for risk assessment step"""
        return f"""<|im_start|>system
You are a risk management expert. Assess the trading risks based on the technical analysis.

<|im_start|>user
Based on this technical analysis:
{tech_analysis}

Assess the risks for trading {market_data['symbol']}:

Current volatility: {market_data['volatility']:.3f}
24h range: ${market_data['low']:.2f} - ${market_data['high']:.2f}

Provide risk assessment covering:
1. Volatility risks
2. Position sizing recommendations
3. Stop loss levels
4. Market timing risks
<|im_end|>

<|im_start|>assistant
"""
    
    def _create_market_context_prompt(self, market_data: Dict, tech_analysis: str, risk_assessment: str) -> str:
        """Create prompt for market context analysis"""
        return f"""<|im_start|>system
You are a market sentiment analyst. Analyze the broader market context and sentiment.

<|im_start|>user
Based on this analysis:
Technical: {tech_analysis}
Risk: {risk_assessment}

Analyze the market context for {market_data['symbol']}:

Market cap: ${market_data['market_cap']:,.0f}
Fear & Greed: {market_data['fear_greed']}
Trend: {market_data['trend']}

Provide market context analysis covering:
1. Market sentiment
2. Broader market trends
3. News and events impact
4. Seasonal factors
<|im_end|>

<|im_start|>assistant
"""
    
    def _create_signal_generation_prompt(self, market_data: Dict, steps: List[TradingStep]) -> str:
        """Create prompt for final signal generation"""
        analysis_summary = "\n".join([f"{step.step_type}: {step.content}" for step in steps])
        
        return f"""<|im_start|>system
You are a trading decision maker. Synthesize all analysis into a final trading decision.

<|im_start|>user
Based on this comprehensive analysis:
{analysis_summary}

Make a final trading decision for {market_data['symbol']}:

Current price: ${market_data['close']:.2f}

Provide:
1. Trading signal (BUY/SELL/HOLD)
2. Confidence level (0-1)
3. Detailed reasoning
4. Position size recommendation
5. Stop loss and take profit levels
<|im_end|>

<|im_start|>assistant
"""
    
    def _create_step_prompt(self, market_data, step):
        """Create prompt for a specific analysis step"""
        base_prompt = f"""<|im_start|>system
You are an expert cryptocurrency trading analyst. Analyze the given market data step by step.

<|im_start|>user
Analyze this {market_data['symbol']} trading data:

Price: ${market_data['close']:.2f} ({market_data['change_24h']:+.2f}%)
Volume: {market_data['volume']:,.0f}
RSI: {market_data['rsi']:.1f}
MACD: {market_data['macd']:.2f}

Current step: {step.step_type}
<|im_end|>

<|im_start|>assistant
{step.content}

Reasoning: {step.reasoning}
Confidence: {step.confidence:.2f}
<|im_end|>"""
        return base_prompt
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using the model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        
        return response
    
    def _extract_signal(self, response: str) -> str:
        """Extract trading signal from response"""
        response_upper = response.upper()
        if "BUY" in response_upper:
            return "BUY"
        elif "SELL" in response_upper:
            return "SELL"
        else:
            return "HOLD"
    
    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from response"""
        # Simple extraction - in practice, you'd use more sophisticated parsing
        lines = response.split('\n')
        reasoning_lines = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ['reasoning', 'because', 'due to', 'based on']):
                reasoning_lines.append(line.strip())
        
        return ' '.join(reasoning_lines) if reasoning_lines else response[:200]
    
    def train_reward_model(self, decisions: List[TradingDecision]):
        """Train the reward model on human feedback"""
        
        dataset = TradingStepDataset(decisions, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.config['rlhf']['batch_size'], shuffle=True)
        
        self.reward_model.train()
        
        for epoch in range(self.config['rlhf']['reward_epochs']):
            total_loss = 0
            
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                rewards = batch['reward'].to(self.device).float()
                
                # Forward pass
                predicted_rewards = self.reward_model(input_ids, attention_mask).squeeze()
                
                # Calculate loss
                loss = self.reward_loss_fn(predicted_rewards, rewards)
                
                # Backward pass
                self.reward_optimizer.zero_grad()
                loss.backward()
                self.reward_optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Reward model epoch {epoch+1}, avg loss: {avg_loss:.4f}")
    
    def train_with_rlhf(self, decisions: List[TradingDecision]):
        """Train the main model using RLHF"""
        
        self.model.train()
        
        for epoch in range(self.config['rlhf']['rl_epochs']):
            total_reward = 0
            total_kl_loss = 0
            
            for decision in decisions:
                # Generate new analysis
                new_decision = self.generate_trading_analysis(decision.market_data)
                
                # Calculate rewards for each step
                for i, step in enumerate(new_decision.steps):
                    # Get reward from reward model
                    prompt = self._create_step_prompt(decision.market_data, step)
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        reward = self.reward_model(inputs['input_ids'], inputs['attention_mask']).item()
                    
                    step.reward = reward
                    total_reward += reward
                
                # Calculate KL divergence loss (to prevent model from deviating too much)
                # This is a simplified version - in practice, you'd compare with reference model
                kl_loss = 0.1  # Placeholder
                total_kl_loss += kl_loss
                
                # Update model based on rewards
                # This is a simplified version - in practice, you'd use PPO or similar
                if total_reward > 0:
                    # Positive reward - reinforce this behavior
                    pass
                else:
                    # Negative reward - discourage this behavior
                    pass
            
            avg_reward = total_reward / len(decisions)
            avg_kl = total_kl_loss / len(decisions)
            logger.info(f"RL epoch {epoch+1}, avg reward: {avg_reward:.4f}, avg KL: {avg_kl:.4f}")
    
    def save_models(self, output_dir: str):
        """Save the trained models"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main model
        self.model.save_pretrained(os.path.join(output_dir, "trading_model"))
        self.tokenizer.save_pretrained(os.path.join(output_dir, "trading_model"))
        
        # Save reward model
        torch.save(self.reward_model.state_dict(), os.path.join(output_dir, "reward_model.pt"))
        
        logger.info(f"Models saved to {output_dir}")

def main():
    """Main training function"""
    
    # Load configuration
    with open("training_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = RLHFTradingTrainer("training_config.yaml")
    
    # Load training data
    with open("training_data/combined_training_data.json", 'r') as f:
        training_data = json.load(f)
    
    # Convert to TradingDecision objects
    decisions = []
    for sample in training_data[:1000]:  # Use first 1000 samples for testing
        decision = TradingDecision(
            market_data=sample['price_data'],
            steps=[],  # Will be generated
            final_signal=sample['signal'],
            final_confidence=sample['confidence'],
            final_reasoning=sample['reasoning']
        )
        decisions.append(decision)
    
    # Train reward model
    logger.info("Training reward model...")
    trainer.train_reward_model(decisions)
    
    # Train with RLHF
    logger.info("Training with RLHF...")
    trainer.train_with_rlhf(decisions)
    
    # Save models
    trainer.save_models("trained_models/rlhf_trading_model")
    
    logger.info("RLHF training completed!")

if __name__ == "__main__":
    main()
