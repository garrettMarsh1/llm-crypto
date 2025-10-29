"""
Quick start script for the crypto trading agent
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from loguru import logger


def print_banner():
    """Print welcome banner"""
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
    print("\n🛠️ Running environment setup...")
    try:
        from scripts.setup_environment import main as setup_main
        setup_main()
    except Exception as e:
        print(f"❌ Setup failed: {e}")


async def run_test():
    """Run system test"""
    print("\n🧪 Running system test...")
    try:
        from scripts.test_system import test_system
        await test_system()
    except Exception as e:
        print(f"❌ Test failed: {e}")


async def run_paper_trading():
    """Start paper trading"""
    print("\n📊 Starting paper trading...")
    print("This will start live paper trading with BTC/USD and ETH/USD")
    print("Press Ctrl+C to stop")
    
    try:
        from agents.trading_agent import CryptoTradingAgent
        agent = CryptoTradingAgent()
        await agent.initialize()
        await agent.start_trading(["BTC/USD", "ETH/USD"])
    except KeyboardInterrupt:
        print("\n🛑 Paper trading stopped")
    except Exception as e:
        print(f"❌ Paper trading failed: {e}")


async def run_backtest():
    """Run backtest"""
    print("\n📈 Running backtest...")
    print("This will backtest the system on historical data")
    
    try:
        from agents.trading_agent import CryptoTradingAgent
        agent = CryptoTradingAgent()
        await agent.initialize()
        
        results = await agent.run_backtest(["BTC/USD", "ETH/USD"], days=7)
        
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"Symbols: {results.get('symbols', [])}")
        print(f"Days: {results.get('days', 0)}")
        print(f"Total Trades: {results.get('performance', {}).get('total_trades', 0)}")
        print(f"Win Rate: {results.get('performance', {}).get('win_rate', 0):.2f}%")
        print(f"Total P&L: ${results.get('performance', {}).get('total_pnl', 0):.2f}")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")


async def run_dashboard():
    """Start monitoring dashboard"""
    print("\n🖥️ Starting monitoring dashboard...")
    print("This will show real-time trading metrics")
    print("Press Ctrl+C to stop")
    
    try:
        from scripts.monitor_dashboard import TradingDashboard
        dashboard = TradingDashboard()
        await dashboard.initialize()
        await dashboard.start_monitoring(refresh_interval=5)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped")
    except Exception as e:
        print(f"❌ Dashboard failed: {e}")


async def run_training():
    """Train the model"""
    print("\n🎯 Starting model training...")
    print("This will fine-tune the LLM for trading")
    
    try:
        from scripts.train_model import main as train_main
        await train_main()
    except Exception as e:
        print(f"❌ Training failed: {e}")


async def run_mcp_server():
    """Run MCP server"""
    print("\n🔧 Starting MCP server...")
    print("This will start the Alpaca MCP server")
    print("Press Ctrl+C to stop")
    
    try:
        from scripts.run_mcp_server import main as mcp_main
        await mcp_main()
    except KeyboardInterrupt:
        print("\n🛑 MCP server stopped")
    except Exception as e:
        print(f"❌ MCP server failed: {e}")


async def main():
    """Main function"""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input().strip()
            
            if choice == "1":
                await run_setup()
            elif choice == "2":
                await run_test()
            elif choice == "3":
                await run_paper_trading()
            elif choice == "4":
                await run_backtest()
            elif choice == "5":
                await run_dashboard()
            elif choice == "6":
                await run_training()
            elif choice == "7":
                await run_mcp_server()
            elif choice == "8":
                print("\nGoodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-8.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        print("\n" + "-" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
