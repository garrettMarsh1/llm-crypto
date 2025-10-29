"""
Main Entry Point for Crypto Trading Agent
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import List, Optional

from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from agents.trading_agent import CryptoTradingAgent
from monitoring.logger import TradingLogger, PerformanceMonitor, HealthChecker


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Crypto Trading Agent")
    parser.add_argument("--mode", choices=["live", "paper", "backtest"], default="paper",
                       help="Trading mode")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USD", "ETH/USD"],
                       help="Trading symbols")
    parser.add_argument("--config", default="config.yaml",
                       help="Configuration file path")
    parser.add_argument("--backtest-days", type=int, default=7,
                       help="Days for backtesting")
    parser.add_argument("--manual-trade", nargs=3, metavar=("SYMBOL", "ACTION", "QUANTITY"),
                       help="Execute manual trade: SYMBOL ACTION QUANTITY")
    
    args = parser.parse_args()
    
    # Initialize logging
    trading_logger = TradingLogger(args.config)
    performance_monitor = PerformanceMonitor(args.config)
    health_checker = HealthChecker(args.config)
    
    try:
        # Initialize trading agent
        agent = CryptoTradingAgent(args.config)
        await agent.initialize()
        
        if args.manual_trade:
            # Execute manual trade
            symbol, action, quantity = args.manual_trade
            result = await agent.manual_trade(symbol, action, float(quantity))
            print(f"Manual trade result: {result}")
            return
        
        if args.mode == "backtest":
            # Run backtest
            logger.info(f"🔄 Running backtest for {args.symbols} over {args.backtest_days} days")
            backtest_results = await agent.run_backtest(args.symbols, args.backtest_days)
            
            print("\n" + "="*50)
            print("BACKTEST RESULTS")
            print("="*50)
            print(f"Symbols: {backtest_results.get('symbols', [])}")
            print(f"Days: {backtest_results.get('days', 0)}")
            print(f"Total Trades: {backtest_results.get('performance', {}).get('total_trades', 0)}")
            print(f"Win Rate: {backtest_results.get('performance', {}).get('win_rate', 0):.2f}%")
            print(f"Total P&L: ${backtest_results.get('performance', {}).get('total_pnl', 0):.2f}")
            print("="*50)
            
            return
        
        # Live/Paper trading
        logger.info(f"🚀 Starting {args.mode} trading for symbols: {args.symbols}")
        
        # Start trading
        await agent.start_trading(args.symbols)
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        if 'agent' in locals():
            await agent.stop_trading()


if __name__ == "__main__":
    asyncio.run(main())
