#!/usr/bin/env python3
"""
Complete RLHF Training Pipeline for Crypto Trading Model
Implements process supervision similar to OpenAI's approach for mathematical reasoning
"""

import os
import json
import yaml
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
from loguru import logger
import argparse
from datetime import datetime
import pandas as pd

from rlhf_trading_trainer import RLHFTradingTrainer, TradingDecision, TradingStep
from step_reward_system import StepRewardSystem, StepType

class RLHFTrainingPipeline:
    """Complete RLHF training pipeline for crypto trading"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.trainer = RLHFTradingTrainer(config_path)
        self.reward_system = StepRewardSystem(self.config)
        
        # Training data
        self.training_data = []
        self.validation_data = []
        
    def load_training_data(self):
        """Load and prepare training data"""
        logger.info("Loading training data...")
        
        # Load existing training data
        with open(self.config['data']['training_data_path'], 'r') as f:
            raw_data = json.load(f)
        
        # Convert to TradingDecision format
        for sample in raw_data[:self.config['data']['rlhf_samples']]:
            decision = TradingDecision(
                market_data=sample['price_data'],
                steps=[],  # Will be generated during training
                final_signal=sample['signal'],
                final_confidence=sample['confidence'],
                final_reasoning=sample['reasoning']
            )
            self.training_data.append(decision)
        
        # Split into training and validation
        split_idx = int(len(self.training_data) * 0.8)
        self.validation_data = self.training_data[split_idx:]
        self.training_data = self.training_data[:split_idx]
        
        logger.info(f"Loaded {len(self.training_data)} training samples")
        logger.info(f"Loaded {len(self.validation_data)} validation samples")
    
    def generate_rlhf_data(self):
        """Generate additional RLHF training data"""
        logger.info("Generating RLHF training data...")
        
        rlhf_data = []
        
        for i, decision in enumerate(self.training_data):
            if i % 100 == 0:
                logger.info(f"Generating RLHF data: {i}/{len(self.training_data)}")
            
            # Generate step-by-step analysis
            new_decision = self.trainer.generate_trading_analysis(decision.market_data)
            
            # Evaluate each step
            for step in new_decision.steps:
                evaluation = self.reward_system.evaluate_step(
                    step_type=StepType(step.step_type),
                    content=step.content,
                    confidence=step.confidence,
                    reasoning=step.reasoning,
                    market_data=decision.market_data,
                    actual_outcome=decision.actual_outcome
                )
                
                # Update step with evaluation results
                step.is_correct = evaluation.correctness
                step.reward = evaluation.reward
            
            rlhf_data.append(new_decision)
        
        self.training_data = rlhf_data
        logger.info(f"Generated {len(rlhf_data)} RLHF training samples")
    
    def train_reward_model(self):
        """Train the reward model on human feedback"""
        logger.info("Training reward model...")
        
        # Prepare training data for reward model
        reward_training_data = []
        
        for decision in self.training_data:
            for step in decision.steps:
                # Create training sample for reward model
                sample = {
                    'market_data': decision.market_data,
                    'step': step,
                    'reward': step.reward,
                    'is_correct': step.is_correct
                }
                reward_training_data.append(sample)
        
        # Train reward model
        self.trainer.train_reward_model(self.training_data)
        
        logger.info("Reward model training completed")
    
    def train_with_rlhf(self):
        """Train the main model using RLHF"""
        logger.info("Starting RLHF training...")
        
        # Train with RLHF
        self.trainer.train_with_rlhf(self.training_data)
        
        logger.info("RLHF training completed")
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        logger.info("Evaluating model...")
        
        evaluation_results = {
            'step_accuracy': {},
            'confidence_calibration': {},
            'reward_correlation': {},
            'trading_performance': {}
        }
        
        # Evaluate on validation data
        total_reward = 0
        step_accuracy = {step_type.value: [] for step_type in StepType}
        
        for decision in self.validation_data:
            # Generate new analysis
            new_decision = self.trainer.generate_trading_analysis(decision.market_data)
            
            # Evaluate each step
            for step in new_decision.steps:
                evaluation = self.reward_system.evaluate_step(
                    step_type=StepType(step.step_type),
                    content=step.content,
                    confidence=step.confidence,
                    reasoning=step.reasoning,
                    market_data=decision.market_data,
                    actual_outcome=decision.actual_outcome
                )
                
                step_accuracy[step.step_type].append(evaluation.correctness)
                total_reward += evaluation.reward
        
        # Calculate metrics
        for step_type, accuracies in step_accuracy.items():
            evaluation_results['step_accuracy'][step_type] = np.mean(accuracies)
        
        evaluation_results['average_reward'] = total_reward / len(self.validation_data)
        
        # Save evaluation results
        with open('rlhf_evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info("Model evaluation completed")
        logger.info(f"Average reward: {evaluation_results['average_reward']:.3f}")
        
        for step_type, accuracy in evaluation_results['step_accuracy'].items():
            logger.info(f"{step_type} accuracy: {accuracy:.3f}")
    
    def save_models(self):
        """Save the trained models"""
        logger.info("Saving models...")
        
        output_dir = self.config['rlhf']['output_dir']
        self.trainer.save_models(output_dir)
        
        # Save reward system
        reward_system_path = os.path.join(output_dir, "reward_system.json")
        with open(reward_system_path, 'w') as f:
            json.dump({
                'step_weights': self.reward_system.step_weights,
                'quality_criteria': self.reward_system.quality_criteria,
                'rewards': self.reward_system.rewards
            }, f, indent=2)
        
        logger.info(f"Models saved to {output_dir}")
    
    def run_training_pipeline(self):
        """Run the complete RLHF training pipeline"""
        logger.info("Starting RLHF training pipeline...")
        
        # Step 1: Load training data
        self.load_training_data()
        
        # Step 2: Generate RLHF data
        self.generate_rlhf_data()
        
        # Step 3: Train reward model
        self.train_reward_model()
        
        # Step 4: Train with RLHF
        self.train_with_rlhf()
        
        # Step 5: Evaluate model
        self.evaluate_model()
        
        # Step 6: Save models
        self.save_models()
        
        logger.info("RLHF training pipeline completed!")

def main():
    parser = argparse.ArgumentParser(description="Run RLHF training for crypto trading model")
    parser.add_argument("--config", type=str, default="rlhf_config.yaml", help="RLHF configuration file")
    parser.add_argument("--data_dir", type=str, default="./training_data", help="Training data directory")
    parser.add_argument("--output_dir", type=str, default="./trained_models/rlhf", help="Output directory")
    parser.add_argument("--samples", type=int, default=5000, help="Number of RLHF samples to generate")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add("logs/rlhf_training_{time}.log")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line arguments
    config['data']['training_data_path'] = os.path.join(args.data_dir, "combined_training_data.json")
    config['rlhf']['output_dir'] = args.output_dir
    config['data']['rlhf_samples'] = args.samples
    
    # Save updated config
    with open(args.config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Initialize and run pipeline
    pipeline = RLHFTrainingPipeline(args.config)
    pipeline.run_training_pipeline()

if __name__ == "__main__":
    main()
