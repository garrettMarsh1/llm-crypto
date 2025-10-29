#!/usr/bin/env python3
"""
Step-by-Step Reward System for Crypto Trading RLHF
Implements process supervision similar to OpenAI's approach for mathematical reasoning
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from loguru import logger

class StepType(Enum):
    TECHNICAL_ANALYSIS = "technical_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    MARKET_CONTEXT = "market_context"
    SIGNAL_GENERATION = "signal_generation"

@dataclass
class StepEvaluation:
    """Evaluation of a single analysis step"""
    step_type: StepType
    content: str
    confidence: float
    reasoning: str
    quality_score: float
    correctness: bool
    reward: float
    feedback: str

class StepRewardSystem:
    """Comprehensive reward system for trading analysis steps"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.step_weights = config['process_supervision']['step_weights']
        self.quality_criteria = config['process_supervision']['quality_criteria']
        self.rewards = config['rewards']
    
    def evaluate_step(self, step_type: StepType, content: str, confidence: float, 
                     reasoning: str, market_data: Dict, actual_outcome: float = None) -> StepEvaluation:
        """Evaluate a single analysis step and calculate reward"""
        
        # Calculate quality score based on step type
        quality_score = self._calculate_quality_score(step_type, content, market_data)
        
        # Determine correctness
        correctness = self._determine_correctness(step_type, content, market_data, actual_outcome)
        
        # Calculate base reward
        base_reward = self._calculate_base_reward(correctness, confidence)
        
        # Add step-specific rewards
        step_rewards = self._calculate_step_specific_rewards(step_type, content, quality_score)
        
        # Add outcome-based rewards
        outcome_rewards = self._calculate_outcome_rewards(step_type, content, actual_outcome)
        
        # Calculate total reward
        total_reward = base_reward + step_rewards + outcome_rewards
        
        # Generate feedback
        feedback = self._generate_feedback(step_type, content, quality_score, correctness)
        
        return StepEvaluation(
            step_type=step_type,
            content=content,
            confidence=confidence,
            reasoning=reasoning,
            quality_score=quality_score,
            correctness=correctness,
            reward=total_reward,
            feedback=feedback
        )
    
    def _calculate_quality_score(self, step_type: StepType, content: str, market_data: Dict) -> float:
        """Calculate quality score based on step-specific criteria"""
        
        criteria = self.quality_criteria[step_type.value]
        score = 0.0
        max_score = len(criteria)
        
        content_lower = content.lower()
        
        if step_type == StepType.TECHNICAL_ANALYSIS:
            # Check for technical analysis quality indicators
            if "rsi" in content_lower and self._mentions_rsi_analysis(content):
                score += 1
            if "macd" in content_lower and self._mentions_macd_analysis(content):
                score += 1
            if any(word in content_lower for word in ["trend", "momentum", "direction"]):
                score += 1
            if any(word in content_lower for word in ["support", "resistance", "level"]):
                score += 1
            if any(word in content_lower for word in ["volume", "liquidity", "activity"]):
                score += 1
            if any(word in content_lower for word in ["bollinger", "band", "channel"]):
                score += 1
        
        elif step_type == StepType.RISK_ASSESSMENT:
            # Check for risk assessment quality indicators
            if any(word in content_lower for word in ["volatility", "risk", "uncertainty"]):
                score += 1
            if any(word in content_lower for word in ["position", "size", "allocation"]):
                score += 1
            if any(word in content_lower for word in ["stop", "loss", "exit"]):
                score += 1
            if any(word in content_lower for word in ["timing", "entry", "exit"]):
                score += 1
            if any(word in content_lower for word in ["correlation", "diversification"]):
                score += 1
            if any(word in content_lower for word in ["scenario", "stress", "test"]):
                score += 1
        
        elif step_type == StepType.MARKET_CONTEXT:
            # Check for market context quality indicators
            if any(word in content_lower for word in ["sentiment", "mood", "feeling"]):
                score += 1
            if any(word in content_lower for word in ["news", "event", "announcement"]):
                score += 1
            if any(word in content_lower for word in ["macro", "economic", "policy"]):
                score += 1
            if any(word in content_lower for word in ["seasonal", "cycle", "pattern"]):
                score += 1
            if any(word in content_lower for word in ["institutional", "retail", "flow"]):
                score += 1
            if any(word in content_lower for word in ["regulatory", "compliance", "legal"]):
                score += 1
        
        elif step_type == StepType.SIGNAL_GENERATION:
            # Check for signal generation quality indicators
            if any(signal in content_lower for signal in ["buy", "sell", "hold"]):
                score += 1
            if any(word in content_lower for word in ["confidence", "probability", "certainty"]):
                score += 1
            if any(word in content_lower for word in ["reasoning", "because", "due to"]):
                score += 1
            if any(word in content_lower for word in ["risk", "management", "protection"]):
                score += 1
            if any(word in content_lower for word in ["target", "profit", "objective"]):
                score += 1
            if any(word in content_lower for word in ["timeframe", "duration", "period"]):
                score += 1
        
        return min(1.0, score / max_score) if max_score > 0 else 0.0
    
    def _mentions_rsi_analysis(self, content: str) -> bool:
        """Check if content mentions RSI analysis"""
        rsi_patterns = [
            r"rsi.*(?:overbought|oversold|neutral)",
            r"(?:overbought|oversold).*rsi",
            r"rsi.*(?:above|below).*\d+",
            r"rsi.*(?:divergence|convergence)"
        ]
        return any(re.search(pattern, content.lower()) for pattern in rsi_patterns)
    
    def _mentions_macd_analysis(self, content: str) -> bool:
        """Check if content mentions MACD analysis"""
        macd_patterns = [
            r"macd.*(?:crossover|cross)",
            r"(?:crossover|cross).*macd",
            r"macd.*(?:above|below).*zero",
            r"macd.*(?:divergence|convergence)"
        ]
        return any(re.search(pattern, content.lower()) for pattern in macd_patterns)
    
    def _determine_correctness(self, step_type: StepType, content: str, 
                             market_data: Dict, actual_outcome: float = None) -> bool:
        """Determine if the step is correct based on market data and outcome"""
        
        if step_type == StepType.TECHNICAL_ANALYSIS:
            return self._evaluate_technical_correctness(content, market_data)
        
        elif step_type == StepType.RISK_ASSESSMENT:
            return self._evaluate_risk_correctness(content, market_data)
        
        elif step_type == StepType.MARKET_CONTEXT:
            return self._evaluate_context_correctness(content, market_data)
        
        elif step_type == StepType.SIGNAL_GENERATION:
            return self._evaluate_signal_correctness(content, market_data, actual_outcome)
        
        return True  # Default to correct if unsure
    
    def _evaluate_technical_correctness(self, content: str, market_data: Dict) -> bool:
        """Evaluate technical analysis correctness"""
        content_lower = content.lower()
        
        # Check RSI analysis
        rsi = market_data.get('rsi', 50)
        if "rsi" in content_lower:
            if rsi > 70 and "oversold" in content_lower:
                return False
            if rsi < 30 and "overbought" in content_lower:
                return False
            if 30 <= rsi <= 70 and ("overbought" in content_lower or "oversold" in content_lower):
                return False
        
        # Check trend analysis
        change_24h = market_data.get('change_24h', 0)
        if change_24h > 0 and "bearish" in content_lower and "trend" in content_lower:
            return False
        if change_24h < 0 and "bullish" in content_lower and "trend" in content_lower:
            return False
        
        return True
    
    def _evaluate_risk_correctness(self, content: str, market_data: Dict) -> bool:
        """Evaluate risk assessment correctness"""
        content_lower = content.lower()
        volatility = market_data.get('volatility', 0.02)
        
        # Check volatility assessment
        if volatility > 0.05 and "low risk" in content_lower:
            return False
        if volatility < 0.01 and "high risk" in content_lower:
            return False
        
        # Check position sizing
        if "large position" in content_lower and volatility > 0.03:
            return False
        if "small position" in content_lower and volatility < 0.01:
            return False
        
        return True
    
    def _evaluate_context_correctness(self, content: str, market_data: Dict) -> bool:
        """Evaluate market context correctness"""
        content_lower = content.lower()
        fear_greed = market_data.get('fear_greed', 'Neutral')
        
        # Check sentiment analysis
        if fear_greed == 'Extreme Fear' and "greed" in content_lower:
            return False
        if fear_greed == 'Extreme Greed' and "fear" in content_lower:
            return False
        
        return True
    
    def _evaluate_signal_correctness(self, content: str, market_data: Dict, actual_outcome: float = None) -> bool:
        """Evaluate signal generation correctness"""
        if actual_outcome is None:
            return True  # Can't evaluate without outcome
        
        content_lower = content.lower()
        signal = self._extract_signal(content)
        
        # Simple correctness based on outcome
        if signal == "BUY" and actual_outcome > 0:
            return True
        elif signal == "SELL" and actual_outcome < 0:
            return True
        elif signal == "HOLD" and abs(actual_outcome) < 0.01:  # Small change
            return True
        else:
            return False
    
    def _extract_signal(self, content: str) -> str:
        """Extract trading signal from content"""
        content_upper = content.upper()
        if "BUY" in content_upper:
            return "BUY"
        elif "SELL" in content_upper:
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_base_reward(self, correctness: bool, confidence: float) -> float:
        """Calculate base reward based on correctness and confidence"""
        if correctness:
            base_reward = self.rewards['correct_step']
            # Bonus for high confidence when correct
            if confidence > 0.7:
                base_reward += self.rewards['high_confidence_correct']
        else:
            base_reward = self.rewards['incorrect_step']
            # Bonus for low confidence when wrong (good calibration)
            if confidence < 0.3:
                base_reward += self.rewards['low_confidence_incorrect']
        
        return base_reward
    
    def _calculate_step_specific_rewards(self, step_type: StepType, content: str, quality_score: float) -> float:
        """Calculate step-specific rewards"""
        rewards = 0.0
        content_lower = content.lower()
        
        if step_type == StepType.TECHNICAL_ANALYSIS:
            if "rsi" in content_lower:
                rewards += self.rewards['technical_analysis']['mentions_rsi']
            if "macd" in content_lower:
                rewards += self.rewards['technical_analysis']['mentions_macd']
            if "bollinger" in content_lower:
                rewards += self.rewards['technical_analysis']['mentions_bollinger']
            if any(word in content_lower for word in ["support", "resistance"]):
                rewards += self.rewards['technical_analysis']['mentions_support_resistance']
        
        elif step_type == StepType.RISK_ASSESSMENT:
            if "volatility" in content_lower:
                rewards += self.rewards['risk_assessment']['mentions_volatility']
            if any(word in content_lower for word in ["stop", "loss"]):
                rewards += self.rewards['risk_assessment']['suggests_stop_loss']
            if any(word in content_lower for word in ["position", "size"]):
                rewards += self.rewards['risk_assessment']['considers_position_size']
            if "timing" in content_lower:
                rewards += self.rewards['risk_assessment']['evaluates_timing']
        
        elif step_type == StepType.MARKET_CONTEXT:
            if "sentiment" in content_lower:
                rewards += self.rewards['market_context']['analyzes_sentiment']
            if "news" in content_lower:
                rewards += self.rewards['market_context']['considers_news']
            if any(word in content_lower for word in ["trend", "cycle"]):
                rewards += self.rewards['market_context']['evaluates_trends']
            if "condition" in content_lower:
                rewards += self.rewards['market_context']['assesses_conditions']
        
        elif step_type == StepType.SIGNAL_GENERATION:
            if any(signal in content_lower for signal in ["buy", "sell", "hold"]):
                rewards += self.rewards['signal_generation']['clear_signal']
            if "confidence" in content_lower:
                rewards += self.rewards['signal_generation']['justified_confidence']
            if any(word in content_lower for word in ["reasoning", "because"]):
                rewards += self.rewards['signal_generation']['includes_reasoning']
            if "risk" in content_lower:
                rewards += self.rewards['signal_generation']['mentions_risk']
        
        # Quality bonus
        rewards += quality_score * 0.1
        
        return rewards
    
    def _calculate_outcome_rewards(self, step_type: StepType, content: str, actual_outcome: float = None) -> float:
        """Calculate outcome-based rewards"""
        if actual_outcome is None:
            return 0.0
        
        rewards = 0.0
        
        # Outcome-based rewards
        if actual_outcome > 0.05:  # Profitable trade
            rewards += self.rewards['profitable_trade']
        elif actual_outcome < -0.05:  # Losing trade
            rewards += self.rewards['losing_trade']
        elif abs(actual_outcome) < 0.01:  # Avoided loss
            rewards += self.rewards['avoided_loss']
        
        return rewards
    
    def _generate_feedback(self, step_type: StepType, content: str, 
                          quality_score: float, correctness: bool) -> str:
        """Generate feedback for the step"""
        
        feedback_parts = []
        
        # Correctness feedback
        if correctness:
            feedback_parts.append("✓ Correct analysis")
        else:
            feedback_parts.append("✗ Incorrect analysis")
        
        # Quality feedback
        if quality_score > 0.8:
            feedback_parts.append("High quality analysis")
        elif quality_score > 0.5:
            feedback_parts.append("Good analysis")
        else:
            feedback_parts.append("Analysis needs improvement")
        
        # Step-specific feedback
        if step_type == StepType.TECHNICAL_ANALYSIS:
            if "rsi" not in content.lower():
                feedback_parts.append("Consider mentioning RSI analysis")
            if "trend" not in content.lower():
                feedback_parts.append("Include trend analysis")
        
        elif step_type == StepType.RISK_ASSESSMENT:
            if "volatility" not in content.lower():
                feedback_parts.append("Consider volatility assessment")
            if "position" not in content.lower():
                feedback_parts.append("Include position sizing")
        
        elif step_type == StepType.MARKET_CONTEXT:
            if "sentiment" not in content.lower():
                feedback_parts.append("Consider market sentiment")
            if "news" not in content.lower():
                feedback_parts.append("Include news analysis")
        
        elif step_type == StepType.SIGNAL_GENERATION:
            if not any(signal in content.upper() for signal in ["BUY", "SELL", "HOLD"]):
                feedback_parts.append("Provide clear trading signal")
            if "confidence" not in content.lower():
                feedback_parts.append("Include confidence level")
        
        return "; ".join(feedback_parts)

def main():
    """Test the step reward system"""
    import yaml
    # Load configuration
    with open("rlhf_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize reward system
    reward_system = StepRewardSystem(config)
    
    # Test data
    market_data = {
        'symbol': 'BTC-USD',
        'close': 45000.0,
        'change_24h': 2.5,
        'rsi': 65.0,
        'volatility': 0.03,
        'fear_greed': 'Greed'
    }
    
    # Test technical analysis step
    tech_content = "RSI is at 65, indicating neutral conditions. MACD shows bullish momentum above zero line. Price is above 20-day moving average, confirming uptrend. Volume is increasing, supporting the bullish move."
    
    evaluation = reward_system.evaluate_step(
        step_type=StepType.TECHNICAL_ANALYSIS,
        content=tech_content,
        confidence=0.8,
        reasoning="Based on multiple technical indicators",
        market_data=market_data,
        actual_outcome=0.03
    )
    
    print(f"Step Type: {evaluation.step_type.value}")
    print(f"Quality Score: {evaluation.quality_score:.2f}")
    print(f"Correctness: {evaluation.correctness}")
    print(f"Reward: {evaluation.reward:.3f}")
    print(f"Feedback: {evaluation.feedback}")

if __name__ == "__main__":
    main()
