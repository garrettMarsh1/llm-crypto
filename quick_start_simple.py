#!/usr/bin/env python3
"""
Crypto Trading Agent - Simple Quick Start (Windows Compatible)
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_banner():
    """Print the application banner"""
    print("=" * 60)
    print("CRYPTO TRADING AGENT - QUICK START")
    print("=" * 60)
    print("AI-powered cryptocurrency trading with LLM integration")
    print("Built with Alpaca API and Model Context Protocol (MCP)")
    print("=" * 60)


def print_menu():
    """Print main menu"""
    print("\nQUICK START OPTIONS:")
    print("1. Setup Environment")
    print("2. Test System")
    print("3. Start Paper Trading")
    print("4. Run Backtest")
    print("5. Open Monitoring Dashboard")
    print("6. Train Model")
    print("7. Run MCP Server")
    print("8. Exit")
    print("\nChoose an option (1-8): ", end="")


async def run_setup():
    """Run environment setup"""
    print("\nRunning environment setup...")
    try:
        from scripts.setup_environment import setup_environment
        await setup_environment()
        print("Setup completed successfully!")
    except Exception as e:
        print(f"Setup failed: {e}")


async def test_system():
    """Test the trading system"""
    print("\nRunning system test...")
    try:
        from scripts.test_system import test_system
        await test_system()
        print("System test completed successfully!")
    except Exception as e:
        print(f"Test failed: {e}")


async def start_paper_trading():
    """Start paper trading"""
    print("\nStarting paper trading...")
    try:
        from main import main
        await main()
    except KeyboardInterrupt:
        print("\nPaper trading stopped")
    except Exception as e:
        print(f"Paper trading failed: {e}")


async def run_backtest():
    """Run backtest"""
    print("\nRunning backtest...")
    try:
        from agents.trading_agent import TradingAgent
        import yaml
        
        # Load config
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Create agent
        agent = TradingAgent(config)
        await agent.initialize()
        
        # Run backtest
        results = await agent.run_backtest(
            symbols=["BTC/USD"],
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        
        print(f"Backtest Results: {results}")
        
    except Exception as e:
        print(f"Backtest failed: {e}")


async def start_dashboard():
    """Start monitoring dashboard"""
    print("\nStarting monitoring dashboard...")
    try:
        from scripts.monitor_dashboard import main as dashboard_main
        await dashboard_main()
    except KeyboardInterrupt:
        print("\nDashboard stopped")
    except Exception as e:
        print(f"Dashboard failed: {e}")


async def start_training():
    """Start model training"""
    print("\nStarting model training...")
    try:
        from scripts.train_model import main as train_main
        await train_main()
    except Exception as e:
        print(f"Training failed: {e}")


async def start_mcp_server():
    """Start MCP server"""
    print("\nStarting MCP server...")
    try:
        from scripts.run_mcp_server import main as mcp_main
        await mcp_main()
    except KeyboardInterrupt:
        print("\nMCP server stopped")
    except Exception as e:
        print(f"MCP server failed: {e}")


async def main():
    """Main function"""
    while True:
        print_banner()
        print_menu()
        
        try:
            choice = input().strip()
            
            if choice == "1":
                await run_setup()
            elif choice == "2":
                await test_system()
            elif choice == "3":
                await start_paper_trading()
            elif choice == "4":
                await run_backtest()
            elif choice == "5":
                await start_dashboard()
            elif choice == "6":
                await start_training()
            elif choice == "7":
                await start_mcp_server()
            elif choice == "8":
                print("\nGoodbye!")
                break
            else:
                print("Invalid choice. Please select 1-8.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
