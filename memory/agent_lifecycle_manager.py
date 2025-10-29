#!/usr/bin/env python3
"""
Agent Lifecycle Manager for Crypto Trading System
Manages agent creation, retirement, and memory transfer
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import threading
import time

from .agent_memory_system import AgentMemorySystem, AgentState

@dataclass
class AgentConfig:
    """Configuration for a trading agent (RTX 4070 optimized)"""
    max_tokens: int = 50000
    max_context_tokens: int = 2048
    max_memory_items: int = 800
    max_lifetime_hours: int = 24
    performance_threshold: float = 0.6
    memory_retention_days: int = 30

@dataclass
class AgentMetrics:
    """Performance metrics for an agent"""
    total_trades: int = 0
    profitable_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_confidence: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    last_updated: datetime = None

class AgentLifecycleManager:
    """Manages the lifecycle of trading agents"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize memory system
        self.memory_system = AgentMemorySystem(self.config)
        
        # Agent configurations
        self.agent_configs: Dict[str, AgentConfig] = {}
        
        # Active agents
        self.active_agents: Dict[str, AgentState] = {}
        
        # Agent metrics
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        
        # Background monitoring
        self.monitoring_thread = None
        self.stop_monitoring = False
        
        # Load existing agents
        self._load_existing_agents()
        
        # Start monitoring
        self.start_monitoring()
    
    def create_agent(self, agent_type: str = "default", custom_config: AgentConfig = None) -> str:
        """Create a new trading agent"""
        
        # Get configuration
        if custom_config:
            config = custom_config
        else:
            config = self._get_agent_config(agent_type)
        
        # Create agent
        agent_id = self.memory_system.create_agent()
        
        # Store configuration
        self.agent_configs[agent_id] = config
        
        # Initialize metrics
        self.agent_metrics[agent_id] = AgentMetrics()
        
        # Add initial memories
        self._add_initial_memories(agent_id, agent_type)
        
        logger.info(f"Created new {agent_type} agent: {agent_id}")
        return agent_id
    
    def get_agent_context(self, agent_id: str, query: str = None) -> Tuple[str, int, bool]:
        """Get context for an agent, returns (context, tokens, is_new_agent)"""
        
        if agent_id not in self.active_agents:
            # Check if agent exists but is retired
            if self._is_agent_retired(agent_id):
                # Create new agent and transfer memories
                new_agent_id = self._create_successor_agent(agent_id)
                context, tokens = self.memory_system.get_agent_context(new_agent_id, query)
                return context, tokens, True
            else:
                raise ValueError(f"Agent {agent_id} not found")
        
        # Check if agent needs retirement
        if self._should_retire_agent(agent_id):
            logger.info(f"Agent {agent_id} needs retirement")
            new_agent_id = self.retire_agent(agent_id)
            context, tokens = self.memory_system.get_agent_context(new_agent_id, query)
            return context, tokens, True
        
        # Get context from current agent
        context, tokens = self.memory_system.get_agent_context(agent_id, query)
        return context, tokens, False
    
    def add_memory(self, agent_id: str, content: str, memory_type: str, 
                   importance: float = 0.5, tags: List[str] = None, 
                   metadata: Dict[str, Any] = None) -> str:
        """Add memory to an agent"""
        
        # Check if agent exists
        if agent_id not in self.active_agents:
            if self._is_agent_retired(agent_id):
                # Add to most recent successor
                successor_id = self._get_latest_successor(agent_id)
                if successor_id:
                    return self.memory_system.add_memory(
                        successor_id, content, memory_type, importance, tags, metadata
                    )
            else:
                raise ValueError(f"Agent {agent_id} not found")
        
        return self.memory_system.add_memory(
            agent_id, content, memory_type, importance, tags, metadata
        )
    
    def update_metrics(self, agent_id: str, metrics: Dict[str, Any]):
        """Update agent performance metrics"""
        
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentMetrics()
        
        # Update metrics
        agent_metrics = self.agent_metrics[agent_id]
        for key, value in metrics.items():
            if hasattr(agent_metrics, key):
                setattr(agent_metrics, key, value)
        
        agent_metrics.last_updated = datetime.now()
        
        # Update in memory system
        self.memory_system.update_performance(agent_id, metrics)
        
        # Check if agent should be retired based on performance
        if self._should_retire_agent(agent_id):
            logger.info(f"Agent {agent_id} performance below threshold, retiring...")
            self.retire_agent(agent_id)
    
    def retire_agent(self, agent_id: str) -> str:
        """Retire an agent and create a successor"""
        
        if agent_id not in self.active_agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Create successor agent
        successor_id = self._create_successor_agent(agent_id)
        
        # Mark original agent as retired
        self.active_agents[agent_id].is_active = False
        
        # Transfer important memories
        self._transfer_memories(agent_id, successor_id)
        
        # Transfer performance insights
        self._transfer_performance_insights(agent_id, successor_id)
        
        logger.info(f"Retired agent {agent_id}, created successor {successor_id}")
        return successor_id
    
    def _create_successor_agent(self, parent_agent_id: str) -> str:
        """Create a successor agent from a parent"""
        
        # Determine agent type from parent
        agent_type = self._get_agent_type(parent_agent_id)
        
        # Create new agent
        successor_id = self.create_agent(agent_type)
        
        # Store parent relationship
        self.memory_system.add_memory(
            agent_id=successor_id,
            content=f"Successor agent created from {parent_agent_id}",
            memory_type="system",
            importance=0.3,
            tags=["successor", "agent_creation"],
            metadata={"parent_agent_id": parent_agent_id}
        )
        
        return successor_id
    
    def _transfer_memories(self, from_agent_id: str, to_agent_id: str):
        """Transfer important memories from one agent to another"""
        
        # Get important memories from parent agent
        important_memories = self.memory_system._get_important_memories(from_agent_id, limit=100)
        
        transferred_count = 0
        for memory in important_memories:
            # Only transfer high-importance memories
            if memory.importance > 0.6:
                self.memory_system.add_memory(
                    agent_id=to_agent_id,
                    content=memory.content,
                    memory_type=memory.memory_type,
                    importance=memory.importance * 0.8,  # Slightly reduce importance
                    tags=memory.tags + ["transferred"],
                    metadata={**memory.metadata, "transferred_from": from_agent_id}
                )
                transferred_count += 1
        
        logger.info(f"Transferred {transferred_count} memories from {from_agent_id} to {to_agent_id}")
    
    def _transfer_performance_insights(self, from_agent_id: str, to_agent_id: str):
        """Transfer performance insights to successor agent"""
        
        if from_agent_id not in self.agent_metrics:
            return
        
        parent_metrics = self.agent_metrics[from_agent_id]
        
        # Create performance summary
        insights = []
        
        if parent_metrics.win_rate > 0:
            insights.append(f"Parent agent achieved {parent_metrics.win_rate:.1%} win rate")
        
        if parent_metrics.total_pnl > 0:
            insights.append(f"Parent agent generated ${parent_metrics.total_pnl:.2f} profit")
        
        if parent_metrics.avg_confidence > 0:
            insights.append(f"Parent agent average confidence: {parent_metrics.avg_confidence:.2f}")
        
        if parent_metrics.max_drawdown > 0:
            insights.append(f"Parent agent max drawdown: {parent_metrics.max_drawdown:.1%}")
        
        if insights:
            insight_content = "Performance insights from parent agent: " + "; ".join(insights)
            self.memory_system.add_memory(
                agent_id=to_agent_id,
                content=insight_content,
                memory_type="performance_insight",
                importance=0.7,
                tags=["performance", "insight", "parent_agent"],
                metadata={"parent_agent_id": from_agent_id}
            )
    
    def _should_retire_agent(self, agent_id: str) -> bool:
        """Check if an agent should be retired"""
        
        if agent_id not in self.active_agents:
            return False
        
        agent = self.active_agents[agent_id]
        config = self.agent_configs.get(agent_id, AgentConfig())
        
        # Check token limit
        if agent.total_tokens_used > config.max_tokens:
            return True
        
        # Check memory limit
        if len(agent.memory_items) > config.max_memory_items:
            return True
        
        # Check lifetime
        lifetime = datetime.now() - agent.created_at
        if lifetime.total_seconds() > config.max_lifetime_hours * 3600:
            return True
        
        # Check performance
        if agent_id in self.agent_metrics:
            metrics = self.agent_metrics[agent_id]
            if metrics.win_rate < config.performance_threshold and metrics.total_trades > 10:
                return True
        
        return False
    
    def _is_agent_retired(self, agent_id: str) -> bool:
        """Check if an agent is retired"""
        return agent_id in self.active_agents and not self.active_agents[agent_id].is_active
    
    def _get_latest_successor(self, agent_id: str) -> Optional[str]:
        """Get the latest successor of a retired agent"""
        # This would require tracking successor relationships
        # For now, return None
        return None
    
    def _get_agent_config(self, agent_type: str) -> AgentConfig:
        """Get configuration for agent type"""
        
        configs = {
            "default": AgentConfig(),
            "conservative": AgentConfig(
                max_tokens=30000,
                max_context_tokens=1536,
                max_memory_items=400,
                performance_threshold=0.7
            ),
            "aggressive": AgentConfig(
                max_tokens=80000,
                max_context_tokens=2560,
                max_memory_items=1200,
                performance_threshold=0.5
            ),
            "research": AgentConfig(
                max_tokens=100000,
                max_context_tokens=3072,
                max_memory_items=1500,
                performance_threshold=0.4
            )
        }
        
        return configs.get(agent_type, configs["default"])
    
    def _add_initial_memories(self, agent_id: str, agent_type: str):
        """Add initial memories to a new agent"""
        
        initial_memories = {
            "default": [
                "Always analyze technical indicators before making trading decisions",
                "Consider risk management and position sizing for every trade",
                "Monitor market sentiment and news events",
                "Keep detailed records of all trading decisions and outcomes"
            ],
            "conservative": [
                "Prioritize capital preservation over aggressive gains",
                "Use smaller position sizes and tighter stop losses",
                "Focus on high-probability, low-risk setups",
                "Avoid trading during high volatility periods"
            ],
            "aggressive": [
                "Look for high-reward opportunities with calculated risks",
                "Use larger position sizes for high-confidence setups",
                "Be willing to take on more volatility for better returns",
                "Act quickly on market opportunities"
            ],
            "research": [
                "Focus on deep market analysis and research",
                "Document all findings and insights thoroughly",
                "Test strategies before implementing them",
                "Continuously learn from market data and patterns"
            ]
        }
        
        memories = initial_memories.get(agent_type, initial_memories["default"])
        
        for memory in memories:
            self.memory_system.add_memory(
                agent_id=agent_id,
                content=memory,
                memory_type="guideline",
                importance=0.8,
                tags=["initial", "guideline", agent_type]
            )
    
    def _load_existing_agents(self):
        """Load existing agents from database"""
        # This would load agents from the database
        # For now, we'll start fresh
        pass
    
    def start_monitoring(self):
        """Start background monitoring of agents"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=self._monitor_agents)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Started agent monitoring")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        logger.info("Stopped agent monitoring")
    
    def _monitor_agents(self):
        """Background monitoring of agents"""
        while not self.stop_monitoring:
            try:
                # Check all active agents
                for agent_id in list(self.active_agents.keys()):
                    if self._should_retire_agent(agent_id):
                        logger.info(f"Monitoring: Agent {agent_id} needs retirement")
                        self.retire_agent(agent_id)
                
                # Cleanup old memories
                self.memory_system.cleanup_old_memories(days=30)
                
                # Sleep for 5 minutes
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in agent monitoring: {e}")
                time.sleep(60)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        
        status = {
            "active_agents": len([a for a in self.active_agents.values() if a.is_active]),
            "total_agents": len(self.active_agents),
            "agents": {}
        }
        
        for agent_id, agent in self.active_agents.items():
            status["agents"][agent_id] = {
                "is_active": agent.is_active,
                "created_at": agent.created_at.isoformat(),
                "last_active": agent.last_active.isoformat(),
                "total_tokens_used": agent.total_tokens_used,
                "memory_count": len(agent.memory_items),
                "performance_metrics": agent.performance_metrics
            }
        
        return status
    
    def close(self):
        """Close the lifecycle manager"""
        self.stop_monitoring()
        self.memory_system.close()

def main():
    """Test the agent lifecycle manager"""
    
    config = {
        'memory': {
            'db_path': './memory/trading_memories.db',
            'max_tokens_per_agent': 100000,
            'max_context_tokens': 4000,
            'compression_threshold': 1000
        }
    }
    
    # Save config
    with open('agent_lifecycle_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Initialize lifecycle manager
    manager = AgentLifecycleManager('agent_lifecycle_config.yaml')
    
    # Create an agent
    agent_id = manager.create_agent("conservative")
    
    # Add some memories
    manager.add_memory(agent_id, "BTC showing strong bullish signals", "analysis", 0.8)
    manager.add_memory(agent_id, "Successful trade: +3.2% profit", "trade", 0.9)
    
    # Update metrics
    manager.update_metrics(agent_id, {
        "total_trades": 5,
        "profitable_trades": 4,
        "win_rate": 0.8,
        "total_pnl": 150.0
    })
    
    # Get context
    context, tokens, is_new = manager.get_agent_context(agent_id, "What should I know about trading?")
    
    print(f"Context ({tokens} tokens, new agent: {is_new}):")
    print(context)
    
    # Get status
    status = manager.get_agent_status()
    print(f"\nAgent status: {status}")
    
    # Close
    manager.close()

if __name__ == "__main__":
    main()
