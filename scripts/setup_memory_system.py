#!/usr/bin/env python3
"""
Setup script for the Memory System
Installs dependencies and initializes the memory system
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path
from loguru import logger

def install_requirements():
    """Install required packages"""
    logger.info("Installing memory system requirements...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "memory_requirements.txt"
        ])
        logger.info("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    logger.info("Creating directories...")
    
    directories = [
        "memory",
        "memory/chroma_db",
        "logs",
        "logs/memory",
        "logs/agents",
        "backups",
        "backups/memory"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def create_config():
    """Create default configuration"""
    logger.info("Creating default configuration...")
    
    config = {
        'memory': {
            'db_path': './memory/trading_memories.db',
            'max_tokens_per_agent': 100000,
            'max_context_tokens': 4000,
            'compression_threshold': 1000,
            'memory_retention_days': 30,
            'importance_threshold': 0.3,
            'max_memories_per_agent': 1000,
            'vector_db': {
                'enabled': True,
                'type': 'chromadb',
                'collection_name': 'trading_memories',
                'embedding_model': 'all-MiniLM-L6-v2',
                'similarity_threshold': 0.7
            }
        },
        'agent_lifecycle': {
            'agent_types': {
                'default': {
                    'max_tokens': 100000,
                    'max_context_tokens': 4000,
                    'max_memory_items': 1000,
                    'max_lifetime_hours': 24,
                    'performance_threshold': 0.6,
                    'memory_retention_days': 30
                },
                'conservative': {
                    'max_tokens': 50000,
                    'max_context_tokens': 2000,
                    'max_memory_items': 500,
                    'max_lifetime_hours': 12,
                    'performance_threshold': 0.7,
                    'memory_retention_days': 45
                },
                'aggressive': {
                    'max_tokens': 150000,
                    'max_context_tokens': 6000,
                    'max_memory_items': 1500,
                    'max_lifetime_hours': 36,
                    'performance_threshold': 0.5,
                    'memory_retention_days': 20
                },
                'research': {
                    'max_tokens': 200000,
                    'max_context_tokens': 8000,
                    'max_memory_items': 2000,
                    'max_lifetime_hours': 48,
                    'performance_threshold': 0.4,
                    'memory_retention_days': 60
                }
            }
        }
    }
    
    with open('memory_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("Configuration created: memory_config.yaml")

def test_installation():
    """Test the installation"""
    logger.info("Testing installation...")
    
    try:
        # Test imports
        import chromadb
        import sentence_transformers
        import sqlite3
        import yaml
        import numpy
        import pandas
        
        logger.info("All imports successful")
        
        # Test memory system
        from memory.agent_memory_system import AgentMemorySystem
        
        config = {
            'memory': {
                'db_path': './memory/test_memories.db',
                'max_tokens_per_agent': 1000,
                'max_context_tokens': 100,
                'compression_threshold': 50
            }
        }
        
        memory_system = AgentMemorySystem(config)
        agent_id = memory_system.create_agent()
        
        # Add test memory
        memory_id = memory_system.add_memory(
            agent_id=agent_id,
            content="Test memory for BTC analysis",
            memory_type="analysis",
            importance=0.8,
            tags=["test", "BTC"]
        )
        
        # Get context
        context, tokens = memory_system.get_agent_context(agent_id, "Test query")
        
        logger.info(f"Test successful: {tokens} tokens in context")
        
        # Cleanup
        memory_system.close()
        os.remove('./memory/test_memories.db')
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("Setting up Memory System for Crypto Trading Agents...")
    
    # Install requirements
    if not install_requirements():
        logger.error("Failed to install requirements")
        return False
    
    # Create directories
    create_directories()
    
    # Create configuration
    create_config()
    
    # Test installation
    if not test_installation():
        logger.error("Installation test failed")
        return False
    
    logger.info("Memory System setup completed successfully!")
    logger.info("You can now use the memory system in your trading agents.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
