#!/usr/bin/env python3
"""
Integrated Trading Agent with Advanced Memory System
Combines memory management, agent lifecycle, and trading logic
"""

import os
import json
import yaml
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from loguru import logger
import torch

from .agent_memory_system import AgentMemorySystem
from .agent_lifecycle_manager import AgentLifecycleManager, AgentConfig
from agents.trading_agent import CryptoTradingAgent

class IntegratedTradingAgent:
    """Trading agent with integrated memory and lifecycle management"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.lifecycle_manager = AgentLifecycleManager(config_path)
        self.memory_system = self.lifecycle_manager.memory_system
        
        # Current agent
        self.current_agent_id: Optional[str] = None
        self.agent_type = "default"
        
        # Trading state
        self.is_trading = False
        self.last_market_update = None
        
        # Performance tracking
        self.session_metrics = {
            'trades_executed': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'start_time': datetime.now()
        }
        
        logger.info("Integrated Trading Agent initialized")
    
    def start_trading(self, agent_type: str = "default") -> str:
        """Start trading with a new agent"""
        
        if self.is_trading:
            logger.warning("Already trading, stopping current session")
            self.stop_trading()
        
        # Create new agent
        self.current_agent_id = self.lifecycle_manager.create_agent(agent_type)
        self.agent_type = agent_type
        
        # Add initial trading memories
        self._add_initial_trading_memories()
        
        self.is_trading = True
        logger.info(f"Started trading with {agent_type} agent: {self.current_agent_id}")
        
        return self.current_agent_id
    
    def stop_trading(self):
        """Stop trading and save session data"""
        
        if not self.is_trading:
            return
        
        # Save session metrics
        self._save_session_metrics()
        
        # Add session summary to memory
        self._add_session_summary()
        
        self.is_trading = False
        logger.info(f"Stopped trading with agent: {self.current_agent_id}")
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data with memory-aware context"""
        
        if not self.is_trading:
            raise ValueError("Not currently trading")
        
        # Get context for analysis
        query = f"Analyze {market_data.get('symbol', 'market')} with current data: {market_data}"
        context, tokens, is_new_agent = self.lifecycle_manager.get_agent_context(
            self.current_agent_id, query
        )
        
        if is_new_agent:
            logger.info(f"Agent retired, now using new agent: {self.current_agent_id}")
        
        # Add current market data to memory
        self._add_market_data_memory(market_data)
        
        # Perform analysis (this would integrate with your existing trading logic)
        analysis = await self._perform_analysis(market_data, context)
        
        # Add analysis to memory
        self._add_analysis_memory(analysis)
        
        return analysis
    
    async def execute_trade(self, trade_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade decision"""
        
        if not self.is_trading:
            raise ValueError("Not currently trading")
        
        # Add trade decision to memory
        self._add_trade_decision_memory(trade_decision)
        
        # Execute trade (integrate with your trading system)
        result = await self._execute_trade_logic(trade_decision)
        
        # Add trade result to memory
        self._add_trade_result_memory(result)
        
        # Update metrics
        self._update_session_metrics(result)
        
        return result
    
    def get_agent_insights(self) -> Dict[str, Any]:
        """Get insights from the current agent's memory"""
        
        if not self.current_agent_id:
            return {}
        
        # Get recent memories
        recent_memories = self.memory_system._get_recent_memories(
            self.current_agent_id, limit=20
        )
        
        # Get important memories
        important_memories = self.memory_system._get_important_memories(
            self.current_agent_id, limit=10
        )
        
        # Analyze patterns
        insights = self._analyze_memory_patterns(recent_memories, important_memories)
        
        return insights
    
    def _add_initial_trading_memories(self):
        """Add initial trading memories to the agent"""
        
        initial_memories = [
            {
                "content": "Trading session started - monitor market conditions carefully",
                "memory_type": "system",
                "importance": 0.6,
                "tags": ["session_start", "trading"]
            },
            {
                "content": "Always verify technical indicators before making trading decisions",
                "memory_type": "guideline",
                "importance": 0.8,
                "tags": ["technical_analysis", "guideline"]
            },
            {
                "content": "Risk management is crucial - never risk more than 2% per trade",
                "memory_type": "guideline",
                "importance": 0.9,
                "tags": ["risk_management", "guideline"]
            }
        ]
        
        for memory in initial_memories:
            self.memory_system.add_memory(
                agent_id=self.current_agent_id,
                content=memory["content"],
                memory_type=memory["memory_type"],
                importance=memory["importance"],
                tags=memory["tags"]
            )
    
    def _add_market_data_memory(self, market_data: Dict[str, Any]):
        """Add market data to memory"""
        
        content = f"Market data for {market_data.get('symbol', 'unknown')}: "
        content += f"Price: ${market_data.get('close', 0):.2f}, "
        content += f"Change: {market_data.get('change_24h', 0):+.2f}%, "
        content += f"Volume: {market_data.get('volume', 0):,.0f}"
        
        self.memory_system.add_memory(
            agent_id=self.current_agent_id,
            content=content,
            memory_type="market_data",
            importance=0.5,
            tags=["market_data", market_data.get('symbol', 'unknown')],
            metadata=market_data
        )
    
    def _add_analysis_memory(self, analysis: Dict[str, Any]):
        """Add analysis to memory"""
        
        content = f"Analysis: {analysis.get('signal', 'HOLD')} signal with "
        content += f"{analysis.get('confidence', 0):.2f} confidence. "
        content += f"Reasoning: {analysis.get('reasoning', 'No reasoning provided')}"
        
        self.memory_system.add_memory(
            agent_id=self.current_agent_id,
            content=content,
            memory_type="analysis",
            importance=0.7,
            tags=["analysis", analysis.get('signal', 'HOLD')],
            metadata=analysis
        )
    
    def _add_trade_decision_memory(self, trade_decision: Dict[str, Any]):
        """Add trade decision to memory"""
        
        content = f"Trade decision: {trade_decision.get('action', 'HOLD')} "
        content += f"{trade_decision.get('symbol', 'unknown')} at "
        content += f"${trade_decision.get('price', 0):.2f} with "
        content += f"{trade_decision.get('quantity', 0)} units"
        
        self.memory_system.add_memory(
            agent_id=self.current_agent_id,
            content=content,
            memory_type="trade_decision",
            importance=0.8,
            tags=["trade_decision", trade_decision.get('symbol', 'unknown')],
            metadata=trade_decision
        )
    
    def _add_trade_result_memory(self, result: Dict[str, Any]):
        """Add trade result to memory"""
        
        pnl = result.get('pnl', 0)
        content = f"Trade result: {result.get('status', 'unknown')} - "
        content += f"P&L: ${pnl:.2f} ({pnl:+.2f}%)"
        
        if result.get('lessons_learned'):
            content += f" Lessons: {result['lessons_learned']}"
        
        self.memory_system.add_memory(
            agent_id=self.current_agent_id,
            content=content,
            memory_type="trade_result",
            importance=0.9 if pnl > 0 else 0.6,
            tags=["trade_result", "profitable" if pnl > 0 else "loss"],
            metadata=result
        )
    
    def _add_session_summary(self):
        """Add session summary to memory"""
        
        duration = datetime.now() - self.session_metrics['start_time']
        win_rate = (self.session_metrics['profitable_trades'] / 
                   max(1, self.session_metrics['trades_executed']))
        
        content = f"Trading session summary: {self.session_metrics['trades_executed']} trades, "
        content += f"{win_rate:.1%} win rate, ${self.session_metrics['total_pnl']:.2f} P&L, "
        content += f"Duration: {duration.total_seconds()/3600:.1f} hours"
        
        self.memory_system.add_memory(
            agent_id=self.current_agent_id,
            content=content,
            memory_type="session_summary",
            importance=0.8,
            tags=["session_summary", "performance"],
            metadata=self.session_metrics
        )
    
    def _save_session_metrics(self):
        """Save session metrics to agent"""
        
        if not self.current_agent_id:
            return
        
        win_rate = (self.session_metrics['profitable_trades'] / 
                   max(1, self.session_metrics['trades_executed']))
        
        metrics = {
            'total_trades': self.session_metrics['trades_executed'],
            'profitable_trades': self.session_metrics['profitable_trades'],
            'win_rate': win_rate,
            'total_pnl': self.session_metrics['total_pnl'],
            'session_duration_hours': (datetime.now() - self.session_metrics['start_time']).total_seconds() / 3600
        }
        
        self.lifecycle_manager.update_metrics(self.current_agent_id, metrics)
    
    def _update_session_metrics(self, result: Dict[str, Any]):
        """Update session metrics with trade result"""
        
        self.session_metrics['trades_executed'] += 1
        
        pnl = result.get('pnl', 0)
        if pnl > 0:
            self.session_metrics['profitable_trades'] += 1
        
        self.session_metrics['total_pnl'] += pnl
    
    async def _perform_analysis(self, market_data: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Perform market analysis (integrate with your existing logic)"""
        
        # This would integrate with your existing trading analysis logic
        # For now, return a mock analysis
        
        analysis = {
            'signal': 'HOLD',
            'confidence': 0.7,
            'reasoning': 'Market conditions are neutral based on technical indicators',
            'technical_indicators': {
                'rsi': market_data.get('rsi', 50),
                'macd': market_data.get('macd', 0),
                'trend': 'neutral'
            },
            'risk_assessment': {
                'volatility': market_data.get('volatility', 0.02),
                'recommended_position_size': 0.02
            },
            'market_context': {
                'sentiment': 'neutral',
                'news_impact': 'minimal'
            }
        }
        
        return analysis
    
    async def _execute_trade_logic(self, trade_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade logic (integrate with your trading system)"""
        
        # This would integrate with your existing trade execution logic
        # For now, return a mock result
        
        result = {
            'status': 'executed',
            'symbol': trade_decision.get('symbol', 'BTC-USD'),
            'action': trade_decision.get('action', 'HOLD'),
            'quantity': trade_decision.get('quantity', 0),
            'price': trade_decision.get('price', 0),
            'pnl': 0.0,  # Would be calculated based on actual execution
            'timestamp': datetime.now().isoformat(),
            'lessons_learned': 'Trade executed successfully'
        }
        
        return result
    
    def _analyze_memory_patterns(self, recent_memories: List, important_memories: List) -> Dict[str, Any]:
        """Analyze patterns in agent's memory"""
        
        insights = {
            'recent_activity': len(recent_memories),
            'important_insights': len(important_memories),
            'patterns': {},
            'recommendations': []
        }
        
        # Analyze signal patterns
        signal_counts = {}
        for memory in recent_memories + important_memories:
            if memory.memory_type == 'analysis':
                signal = memory.metadata.get('signal', 'HOLD')
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        insights['patterns']['signal_distribution'] = signal_counts
        
        # Analyze performance patterns
        profitable_trades = 0
        total_trades = 0
        for memory in recent_memories + important_memories:
            if memory.memory_type == 'trade_result':
                total_trades += 1
                if memory.metadata.get('pnl', 0) > 0:
                    profitable_trades += 1
        
        if total_trades > 0:
            insights['patterns']['win_rate'] = profitable_trades / total_trades
        
        # Generate recommendations
        if insights['patterns'].get('win_rate', 0) < 0.5:
            insights['recommendations'].append("Consider more conservative trading approach")
        
        if signal_counts.get('BUY', 0) > signal_counts.get('SELL', 0) * 2:
            insights['recommendations'].append("Consider more balanced signal distribution")
        
        return insights
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        
        if not self.current_agent_id:
            return {'status': 'not_trading'}
        
        agent_stats = self.memory_system.get_agent_stats(self.current_agent_id)
        insights = self.get_agent_insights()
        
        return {
            'status': 'trading' if self.is_trading else 'idle',
            'agent_id': self.current_agent_id,
            'agent_type': self.agent_type,
            'agent_stats': agent_stats,
            'insights': insights,
            'session_metrics': self.session_metrics
        }
    
    def close(self):
        """Close the integrated trading agent"""
        
        if self.is_trading:
            self.stop_trading()
        
        self.lifecycle_manager.close()

def main():
    """Test the integrated trading agent"""
    
    # Create config
    config = {
        'memory': {
            'db_path': './memory/trading_memories.db',
            'max_tokens_per_agent': 100000,
            'max_context_tokens': 4000,
            'compression_threshold': 1000
        }
    }
    
    with open('integrated_agent_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Initialize agent
    agent = IntegratedTradingAgent('integrated_agent_config.yaml')
    
    # Start trading
    agent_id = agent.start_trading("conservative")
    print(f"Started trading with agent: {agent_id}")
    
    # Simulate market analysis
    market_data = {
        'symbol': 'BTC-USD',
        'close': 45000,
        'change_24h': 2.5,
        'rsi': 65,
        'macd': 150,
        'volume': 1000000
    }
    
    # Run analysis
    async def test_analysis():
        analysis = await agent.analyze_market(market_data)
        print(f"Analysis: {analysis}")
        
        # Get insights
        insights = agent.get_agent_insights()
        print(f"Insights: {insights}")
        
        # Get status
        status = agent.get_agent_status()
        print(f"Status: {status}")
    
    # Run test
    asyncio.run(test_analysis())
    
    # Stop trading
    agent.stop_trading()
    agent.close()

if __name__ == "__main__":
    main()
