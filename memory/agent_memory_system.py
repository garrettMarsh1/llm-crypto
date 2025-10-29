#!/usr/bin/env python3
"""
Advanced Memory System for Crypto Trading Agent
Implements efficient token management, semantic memory, and agent lifecycle management
"""

import os
import json
import pickle
import numpy as np
import sqlite3
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import zlib
from loguru import logger

# Vector database for semantic search
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available. Install with: pip install chromadb")

# Text processing
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not available. Install with: pip install sentence-transformers")

@dataclass
class MemoryItem:
    """Individual memory item"""
    id: str
    content: str
    memory_type: str  # 'trade', 'analysis', 'market_event', 'strategy', 'lesson'
    timestamp: datetime
    importance: float  # 0-1, how important this memory is
    tags: List[str]
    metadata: Dict[str, Any]
    token_count: int
    compressed_content: Optional[bytes] = None

@dataclass
class AgentState:
    """Current state of an agent"""
    agent_id: str
    created_at: datetime
    last_active: datetime
    total_tokens_used: int
    context_tokens: int
    memory_items: List[str]  # IDs of loaded memory items
    performance_metrics: Dict[str, float]
    is_active: bool = True

class AgentMemorySystem:
    """Advanced memory system for crypto trading agents"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.db_path = config['memory']['db_path']
        self.max_tokens_per_agent = config['memory']['max_tokens_per_agent']
        self.max_context_tokens = config['memory']['max_context_tokens']
        self.compression_threshold = config['memory']['compression_threshold']
        
        # Initialize database
        self._init_database()
        
        # Initialize vector database if available
        if CHROMADB_AVAILABLE:
            self._init_vector_db()
        else:
            self.vector_db = None
        
        # Initialize embedding model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.embedding_model = None
        
        # Active agents
        self.active_agents: Dict[str, AgentState] = {}
        
        # Memory cache
        self.memory_cache: Dict[str, MemoryItem] = {}
        
    def _init_database(self):
        """Initialize SQLite database for memory storage"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_items (
                id TEXT PRIMARY KEY,
                content TEXT,
                memory_type TEXT,
                timestamp TEXT,
                importance REAL,
                tags TEXT,
                metadata TEXT,
                token_count INTEGER,
                compressed_content BLOB
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_states (
                agent_id TEXT PRIMARY KEY,
                created_at TEXT,
                last_active TEXT,
                total_tokens_used INTEGER,
                context_tokens INTEGER,
                memory_items TEXT,
                performance_metrics TEXT,
                is_active BOOLEAN
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id TEXT PRIMARY KEY,
                embedding BLOB,
                FOREIGN KEY (memory_id) REFERENCES memory_items (id)
            )
        ''')
        
        self.conn.commit()
    
    def _init_vector_db(self):
        """Initialize ChromaDB for semantic search"""
        try:
            self.vector_db = chromadb.PersistentClient(
                path=os.path.join(os.path.dirname(self.db_path), "chroma_db")
            )
            self.collection = self.vector_db.get_or_create_collection(
                name="trading_memories",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.vector_db = None
    
    def create_agent(self, agent_id: str = None) -> str:
        """Create a new trading agent"""
        if agent_id is None:
            agent_id = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        agent_state = AgentState(
            agent_id=agent_id,
            created_at=datetime.now(),
            last_active=datetime.now(),
            total_tokens_used=0,
            context_tokens=0,
            memory_items=[],
            performance_metrics={}
        )
        
        self.active_agents[agent_id] = agent_state
        self._save_agent_state(agent_state)
        
        logger.info(f"Created new agent: {agent_id}")
        return agent_id
    
    def get_agent_context(self, agent_id: str, query: str = None) -> Tuple[str, int]:
        """Get context for an agent, optimized for token usage"""
        
        if agent_id not in self.active_agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.active_agents[agent_id]
        
        # Check if agent needs to be retired
        if agent.total_tokens_used > self.max_tokens_per_agent:
            logger.warning(f"Agent {agent_id} has reached token limit, retiring...")
            self.retire_agent(agent_id)
            return self._get_retired_agent_context(agent_id, query)
        
        # Get relevant memories
        relevant_memories = self._get_relevant_memories(agent_id, query)
        
        # Build context
        context_parts = []
        total_tokens = 0
        
        # Add recent memories
        recent_memories = self._get_recent_memories(agent_id, limit=10)
        for memory in recent_memories:
            if total_tokens + memory.token_count > self.max_context_tokens:
                break
            context_parts.append(f"[{memory.memory_type.upper()}] {memory.content}")
            total_tokens += memory.token_count
        
        # Add relevant memories
        for memory in relevant_memories:
            if total_tokens + memory.token_count > self.max_context_tokens:
                break
            if memory.id not in [m.id for m in recent_memories]:
                context_parts.append(f"[{memory.memory_type.upper()}] {memory.content}")
                total_tokens += memory.token_count
        
        # Add performance summary
        perf_summary = self._get_performance_summary(agent_id)
        if total_tokens + len(perf_summary.split()) < self.max_context_tokens:
            context_parts.append(f"[PERFORMANCE] {perf_summary}")
            total_tokens += len(perf_summary.split())
        
        context = "\n\n".join(context_parts)
        
        # Update agent state
        agent.context_tokens = total_tokens
        agent.last_active = datetime.now()
        self._save_agent_state(agent)
        
        return context, total_tokens
    
    def _get_relevant_memories(self, agent_id: str, query: str = None) -> List[MemoryItem]:
        """Get memories relevant to the query using semantic search"""
        
        if not query or not self.vector_db or not self.embedding_model:
            # Fallback to importance-based selection
            return self._get_important_memories(agent_id)
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Search vector database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=20,
                where={"agent_id": agent_id}
            )
            
            # Get memory items
            memory_ids = results['ids'][0]
            memories = []
            for memory_id in memory_ids:
                if memory_id in self.memory_cache:
                    memories.append(self.memory_cache[memory_id])
                else:
                    memory = self._load_memory_item(memory_id)
                    if memory:
                        self.memory_cache[memory_id] = memory
                        memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return self._get_important_memories(agent_id)
    
    def _get_important_memories(self, agent_id: str, limit: int = 20) -> List[MemoryItem]:
        """Get most important memories for an agent"""
        
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id FROM memory_items 
            WHERE metadata LIKE ? 
            ORDER BY importance DESC, timestamp DESC 
            LIMIT ?
        ''', (f'%"agent_id": "{agent_id}"%', limit))
        
        memory_ids = [row[0] for row in cursor.fetchall()]
        memories = []
        
        for memory_id in memory_ids:
            if memory_id in self.memory_cache:
                memories.append(self.memory_cache[memory_id])
            else:
                memory = self._load_memory_item(memory_id)
                if memory:
                    self.memory_cache[memory_id] = memory
                    memories.append(memory)
        
        return memories
    
    def _get_recent_memories(self, agent_id: str, limit: int = 10) -> List[MemoryItem]:
        """Get most recent memories for an agent"""
        
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id FROM memory_items 
            WHERE metadata LIKE ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (f'%"agent_id": "{agent_id}"%', limit))
        
        memory_ids = [row[0] for row in cursor.fetchall()]
        memories = []
        
        for memory_id in memory_ids:
            if memory_id in self.memory_cache:
                memories.append(self.memory_cache[memory_id])
            else:
                memory = self._load_memory_item(memory_id)
                if memory:
                    self.memory_cache[memory_id] = memory
                    memories.append(memory)
        
        return memories
    
    def _get_performance_summary(self, agent_id: str) -> str:
        """Get performance summary for an agent"""
        
        agent = self.active_agents[agent_id]
        metrics = agent.performance_metrics
        
        summary_parts = []
        
        if 'total_trades' in metrics:
            summary_parts.append(f"Total trades: {metrics['total_trades']}")
        
        if 'win_rate' in metrics:
            summary_parts.append(f"Win rate: {metrics['win_rate']:.1%}")
        
        if 'total_pnl' in metrics:
            summary_parts.append(f"Total P&L: ${metrics['total_pnl']:.2f}")
        
        if 'avg_confidence' in metrics:
            summary_parts.append(f"Avg confidence: {metrics['avg_confidence']:.2f}")
        
        return ", ".join(summary_parts) if summary_parts else "No performance data available"
    
    def add_memory(self, agent_id: str, content: str, memory_type: str, 
                   importance: float = 0.5, tags: List[str] = None, 
                   metadata: Dict[str, Any] = None) -> str:
        """Add a new memory item"""
        
        if agent_id not in self.active_agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Generate memory ID
        memory_id = hashlib.md5(f"{agent_id}_{content}_{datetime.now()}".encode()).hexdigest()
        
        # Calculate token count
        token_count = len(content.split())
        
        # Compress content if needed
        compressed_content = None
        if token_count > self.compression_threshold:
            compressed_content = zlib.compress(content.encode())
        
        # Create memory item
        memory_item = MemoryItem(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            timestamp=datetime.now(),
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
            token_count=token_count,
            compressed_content=compressed_content
        )
        
        # Add agent_id to metadata
        memory_item.metadata['agent_id'] = agent_id
        
        # Save to database
        self._save_memory_item(memory_item)
        
        # Add to vector database if available
        if self.vector_db and self.embedding_model:
            try:
                embedding = self.embedding_model.encode([content])[0].tolist()
                self.collection.add(
                    ids=[memory_id],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[memory_item.metadata]
                )
            except Exception as e:
                logger.error(f"Error adding to vector database: {e}")
        
        # Update agent state
        agent = self.active_agents[agent_id]
        agent.total_tokens_used += token_count
        agent.memory_items.append(memory_id)
        
        # Cache memory
        self.memory_cache[memory_id] = memory_item
        
        logger.info(f"Added memory {memory_id} for agent {agent_id}")
        return memory_id
    
    def _save_memory_item(self, memory_item: MemoryItem):
        """Save memory item to database"""
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO memory_items 
            (id, content, memory_type, timestamp, importance, tags, metadata, token_count, compressed_content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory_item.id,
            memory_item.content,
            memory_item.memory_type,
            memory_item.timestamp.isoformat(),
            memory_item.importance,
            json.dumps(memory_item.tags),
            json.dumps(memory_item.metadata),
            memory_item.token_count,
            memory_item.compressed_content
        ))
        self.conn.commit()
    
    def _load_memory_item(self, memory_id: str) -> Optional[MemoryItem]:
        """Load memory item from database"""
        
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT content, memory_type, timestamp, importance, tags, metadata, token_count, compressed_content
            FROM memory_items WHERE id = ?
        ''', (memory_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        content, memory_type, timestamp, importance, tags, metadata, token_count, compressed_content = row
        
        # Decompress if needed
        if compressed_content:
            content = zlib.decompress(compressed_content).decode()
        
        return MemoryItem(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            timestamp=datetime.fromisoformat(timestamp),
            importance=importance,
            tags=json.loads(tags),
            metadata=json.loads(metadata),
            token_count=token_count,
            compressed_content=compressed_content
        )
    
    def _save_agent_state(self, agent_state: AgentState):
        """Save agent state to database"""
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO agent_states 
            (agent_id, created_at, last_active, total_tokens_used, context_tokens, 
             memory_items, performance_metrics, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            agent_state.agent_id,
            agent_state.created_at.isoformat(),
            agent_state.last_active.isoformat(),
            agent_state.total_tokens_used,
            agent_state.context_tokens,
            json.dumps(agent_state.memory_items),
            json.dumps(agent_state.performance_metrics),
            agent_state.is_active
        ))
        self.conn.commit()
    
    def retire_agent(self, agent_id: str):
        """Retire an agent and create a new one"""
        
        if agent_id not in self.active_agents:
            return
        
        agent = self.active_agents[agent_id]
        agent.is_active = False
        self._save_agent_state(agent)
        
        # Create new agent
        new_agent_id = self.create_agent()
        
        # Transfer important memories to new agent
        important_memories = self._get_important_memories(agent_id, limit=50)
        for memory in important_memories:
            # Create new memory for new agent
            new_memory_id = self.add_memory(
                agent_id=new_agent_id,
                content=memory.content,
                memory_type=memory.memory_type,
                importance=memory.importance,
                tags=memory.tags,
                metadata=memory.metadata
            )
        
        logger.info(f"Retired agent {agent_id}, created new agent {new_agent_id}")
        return new_agent_id
    
    def _get_retired_agent_context(self, agent_id: str, query: str = None) -> Tuple[str, int]:
        """Get context from retired agent"""
        
        # Get performance summary
        perf_summary = self._get_performance_summary(agent_id)
        
        # Get important lessons learned
        important_memories = self._get_important_memories(agent_id, limit=20)
        context_parts = [f"[RETIRED_AGENT] {perf_summary}"]
        
        total_tokens = len(perf_summary.split())
        
        for memory in important_memories:
            if total_tokens + memory.token_count > self.max_context_tokens:
                break
            context_parts.append(f"[{memory.memory_type.upper()}] {memory.content}")
            total_tokens += memory.token_count
        
        context = "\n\n".join(context_parts)
        return context, total_tokens
    
    def update_performance(self, agent_id: str, metrics: Dict[str, float]):
        """Update agent performance metrics"""
        
        if agent_id not in self.active_agents:
            return
        
        agent = self.active_agents[agent_id]
        agent.performance_metrics.update(metrics)
        self._save_agent_state(agent)
    
    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get agent statistics"""
        
        if agent_id not in self.active_agents:
            return {}
        
        agent = self.active_agents[agent_id]
        
        return {
            'agent_id': agent.agent_id,
            'created_at': agent.created_at.isoformat(),
            'last_active': agent.last_active.isoformat(),
            'total_tokens_used': agent.total_tokens_used,
            'context_tokens': agent.context_tokens,
            'memory_count': len(agent.memory_items),
            'performance_metrics': agent.performance_metrics,
            'is_active': agent.is_active
        }
    
    def cleanup_old_memories(self, days: int = 30):
        """Clean up old, unimportant memories"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cursor = self.conn.cursor()
        cursor.execute('''
            DELETE FROM memory_items 
            WHERE timestamp < ? AND importance < 0.3
        ''', (cutoff_date.isoformat(),))
        
        deleted_count = cursor.rowcount
        self.conn.commit()
        
        logger.info(f"Cleaned up {deleted_count} old memories")
        return deleted_count
    
    def close(self):
        """Close the memory system"""
        if hasattr(self, 'conn'):
            self.conn.close()
        
        if self.vector_db:
            # ChromaDB handles its own cleanup
            pass

def main():
    """Test the memory system"""
    
    config = {
        'memory': {
            'db_path': './memory/trading_memories.db',
            'max_tokens_per_agent': 100000,
            'max_context_tokens': 4000,
            'compression_threshold': 1000
        }
    }
    
    # Initialize memory system
    memory_system = AgentMemorySystem(config)
    
    # Create an agent
    agent_id = memory_system.create_agent()
    
    # Add some memories
    memory_system.add_memory(
        agent_id=agent_id,
        content="BTC showed strong bullish momentum with RSI at 65 and MACD above zero line",
        memory_type="analysis",
        importance=0.8,
        tags=["BTC", "technical_analysis", "bullish"]
    )
    
    memory_system.add_memory(
        agent_id=agent_id,
        content="Successful trade: Bought BTC at $45,000, sold at $46,500 for 3.3% profit",
        memory_type="trade",
        importance=0.9,
        tags=["BTC", "profitable_trade", "3.3%_profit"]
    )
    
    # Get context
    context, tokens = memory_system.get_agent_context(agent_id, "What should I know about BTC trading?")
    
    print(f"Context ({tokens} tokens):")
    print(context)
    
    # Get agent stats
    stats = memory_system.get_agent_stats(agent_id)
    print(f"\nAgent stats: {stats}")
    
    # Close
    memory_system.close()

if __name__ == "__main__":
    main()
