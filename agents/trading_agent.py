"""
Main Crypto Trading Agent with LLM Integration
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yaml
from pathlib import Path
from dotenv import load_dotenv

from loguru import logger

# Load environment variables
load_dotenv()

# Import our modules
from models.trading_model import TradingModel
from data.market_data import MarketDataStreamer, MarketDataProcessor
from risk.risk_manager import RiskManager
from mcp_servers.alpaca_server import AlpacaMCPServer


class CryptoTradingAgent:
    """Main crypto trading agent that integrates LLM, market data, and risk management"""
    
    def __init__(self, config_path = "config.yaml"):
        # Handle both file path and config dict
        if isinstance(config_path, dict):
            self.config = config_path
        else:
            self.config = self._load_config(config_path)
        self.model = None
        self.data_streamer = None
        self.data_processor = None
        self.risk_manager = None
        self.alpaca_mcp = None
        self.running = False
        self.positions = {}
        self.orders = {}
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "start_time": datetime.now()
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file and substitute environment variables"""
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Substitute environment variables
        config_content = os.path.expandvars(config_content)
        
        return yaml.safe_load(config_content)
    
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("🚀 Initializing Crypto Trading Agent...")
            
            # Initialize model
            logger.info("Loading LLM model...")
            # TradingModel expects a config file path, not a dict
            # For now, we'll create a temporary config file or use a default path
            config_path = "config.yaml"
            if isinstance(self.config, dict):
                # If config is a dict, we need to save it to a file first
                import yaml
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f)
            
            self.model = TradingModel(config_path)
            self.model.load_model()
            self.model.setup_lora()
            
            # Initialize market data
            logger.info("Setting up market data pipeline...")
            # MarketDataStreamer also expects a config file path
            self.data_streamer = MarketDataStreamer(config_path)
            await self.data_streamer.initialize()
            
            self.data_processor = MarketDataProcessor(self.data_streamer.buffer)
            self.data_streamer.add_callback(self.data_processor.process_data_update)
            
            # Initialize risk manager
            logger.info("Setting up risk management...")
            self.risk_manager = RiskManager(config_path)
            
            # Get initial account data and set it in risk manager
            try:
                account_info = await self.alpaca_mcp._get_account_info()
                if account_info and len(account_info) > 0:
                    account_data = json.loads(account_info[0].text)
                    account_value = float(account_data.get("equity", 10000))
                    self.risk_manager.set_account_value(account_value)
                    logger.info(f"✅ Account value set: ${account_value:,.2f}")
            except Exception as e:
                logger.warning(f"⚠️ Could not get account info: {e}")
                self.risk_manager.set_account_value(10000)  # Fallback
            
            # Initialize Alpaca MCP server
            logger.info("Setting up Alpaca MCP server...")
            self.alpaca_mcp = AlpacaMCPServer(config_path)
            await self.alpaca_mcp.initialize()
            
            # Add trading agent as callback for data updates
            self.data_streamer.add_callback(self._handle_market_data)
            
            logger.info("✅ Crypto Trading Agent initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize trading agent: {e}")
            raise
    
    async def _handle_market_data(self, data_type: str, data: Dict[str, Any]):
        """Handle incoming market data updates"""
        try:
            symbol = data.get("symbol")
            if not symbol:
                return
            
            # Only process bars for trading decisions
            if data_type != "bar":
                return
            
            # Get comprehensive market context
            market_context = await self.data_streamer.get_market_context(symbol)
            
            # Generate trading signal
            signal = self.model.generate_signal(market_context)
            
            # Process signal through risk management
            if await self.risk_manager.validate_signal(signal, self.positions):
                await self._execute_trading_signal(signal, market_context)
            else:
                logger.info(f"⚠️ Signal rejected by risk manager for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error handling market data: {e}")
    
    async def _execute_trading_signal(self, signal: Dict[str, Any], market_context: Dict[str, Any]):
        """Execute trading signal through Alpaca MCP"""
        try:
            symbol = signal["symbol"]
            action = signal["action"]
            confidence = signal["confidence"]
            
            if action == "hold":
                return
            
            # Calculate position size based on risk management
            position_size = await self.risk_manager.calculate_position_size(
                symbol, signal, market_context, self.positions
            )
            
            if position_size <= 0:
                logger.info(f"⚠️ Position size too small for {symbol}: {position_size}")
                return
            
            # Get current price
            current_price = market_context.get("latest_price")
            if not current_price:
                logger.error(f"❌ No current price available for {symbol}")
                return
            
            # Prepare order parameters
            side = "buy" if action == "buy" else "sell"
            quantity = position_size / current_price  # Convert to crypto units
            
            # Place order through MCP server
            if self.config['trading']['order_type'] == "market":
                order_result = await self.alpaca_mcp._place_market_order({
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity
                })
            else:  # limit order
                limit_price = current_price * (1.001 if action == "buy" else 0.999)  # Slight buffer
                order_result = await self.alpaca_mcp._place_limit_order({
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "limit_price": limit_price
                })
            
            # Log the trade
            self._log_trade(signal, market_context, order_result, quantity)
            
            # Update performance metrics
            self._update_performance_metrics(signal, order_result)
            
        except Exception as e:
            logger.error(f"❌ Error executing trading signal: {e}")
    
    def _log_trade(self, signal: Dict[str, Any], market_context: Dict[str, Any], 
                   order_result: List, quantity: float):
        """Log trade details"""
        try:
            trade_log = {
                "timestamp": datetime.now().isoformat(),
                "symbol": signal["symbol"],
                "action": signal["action"],
                "confidence": signal["confidence"],
                "reasoning": signal["reasoning"],
                "quantity": quantity,
                "price": market_context.get("latest_price"),
                "order_result": order_result[0].text if order_result else "Failed",
                "technical_indicators": market_context.get("technical_indicators", {})
            }
            
            logger.info(f"📊 Trade executed: {json.dumps(trade_log, indent=2)}")
            
            # Save to file
            log_file = Path("logs/trades.jsonl")
            log_file.parent.mkdir(exist_ok=True)
            
            with open(log_file, "a") as f:
                f.write(json.dumps(trade_log) + "\n")
                
        except Exception as e:
            logger.error(f"❌ Error logging trade: {e}")
    
    def _update_performance_metrics(self, signal: Dict[str, Any], order_result: List):
        """Update performance metrics"""
        try:
            self.performance_metrics["total_trades"] += 1
            
            # Track actual trade execution
            if signal["action"] in ["buy", "sell"] and order_result:
                # Parse order result to get actual trade details
                try:
                    order_data = json.loads(order_result[0].text) if hasattr(order_result[0], 'text') else order_result[0]
                    
                    # Track order status
                    if order_data.get("status") == "filled":
                        self.performance_metrics["filled_trades"] = self.performance_metrics.get("filled_trades", 0) + 1
                    elif order_data.get("status") == "rejected":
                        self.performance_metrics["rejected_trades"] = self.performance_metrics.get("rejected_trades", 0) + 1
                    
                    # Track by signal confidence (temporary until we have actual P&L)
                    if signal["confidence"] > 0.7:
                        self.performance_metrics["high_confidence_trades"] = self.performance_metrics.get("high_confidence_trades", 0) + 1
                    else:
                        self.performance_metrics["low_confidence_trades"] = self.performance_metrics.get("low_confidence_trades", 0) + 1
                        
                except Exception as parse_error:
                    logger.error(f"❌ Error parsing order result: {parse_error}")
                    # Fallback to simple counting
                    if signal["confidence"] > 0.7:
                        self.performance_metrics["winning_trades"] += 1
                    else:
                        self.performance_metrics["losing_trades"] += 1
            
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
    
    async def start_trading(self, symbols: List[str] = None):
        """Start the trading agent"""
        try:
            if symbols is None:
                symbols = self.config['trading']['crypto_pairs']
            
            logger.info(f"🎯 Starting trading for symbols: {symbols}")
            
            self.running = True
            
            # Start data streaming
            await self.data_streamer.start_streaming(symbols)
            
        except Exception as e:
            logger.error(f"❌ Error starting trading: {e}")
            self.running = False
            raise
    
    async def stop_trading(self):
        """Stop the trading agent"""
        try:
            logger.info("🛑 Stopping trading agent...")
            
            self.running = False
            
            # Stop data streaming
            await self.data_streamer.stop_streaming()
            
            # Print final performance metrics
            self._print_performance_summary()
            
            logger.info("✅ Trading agent stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping trading agent: {e}")
    
    def _print_performance_summary(self):
        """Print performance summary"""
        try:
            duration = datetime.now() - self.performance_metrics["start_time"]
            
            win_rate = (self.performance_metrics["winning_trades"] / 
                       max(1, self.performance_metrics["total_trades"])) * 100
            
            logger.info("📈 Performance Summary:")
            logger.info(f"  Duration: {duration}")
            logger.info(f"  Total Trades: {self.performance_metrics['total_trades']}")
            logger.info(f"  Winning Trades: {self.performance_metrics['winning_trades']}")
            logger.info(f"  Losing Trades: {self.performance_metrics['losing_trades']}")
            logger.info(f"  Win Rate: {win_rate:.2f}%")
            logger.info(f"  Total P&L: ${self.performance_metrics['total_pnl']:.2f}")
            logger.info(f"  Max Drawdown: {self.performance_metrics['max_drawdown']:.2f}%")
            
        except Exception as e:
            logger.error(f"❌ Error printing performance summary: {e}")
    
    async def get_account_status(self) -> Dict[str, Any]:
        """Get current account status"""
        try:
            account_info = await self.alpaca_mcp._get_account_info()
            positions = await self.alpaca_mcp._get_positions()
            orders = await self.alpaca_mcp._get_orders({"status": "open", "limit": 10})
            
            return {
                "account": json.loads(account_info[0].text) if account_info else {},
                "positions": json.loads(positions[0].text) if positions else [],
                "open_orders": json.loads(orders[0].text) if orders else [],
                "performance_metrics": self.performance_metrics,
                "model_info": self.model.get_model_info() if self.model else {}
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting account status: {e}")
            return {"error": str(e)}
    
    async def manual_trade(self, symbol: str, action: str, quantity: float) -> Dict[str, Any]:
        """Execute manual trade"""
        try:
            if action not in ["buy", "sell"]:
                return {"error": "Invalid action. Use 'buy' or 'sell'"}
            
            # Get current price
            market_context = await self.data_streamer.get_market_context(symbol)
            current_price = market_context.get("latest_price")
            
            if not current_price:
                return {"error": "No current price available"}
            
            # Place order
            if self.config['trading']['order_type'] == "market":
                order_result = await self.alpaca_mcp._place_market_order({
                    "symbol": symbol,
                    "side": action,
                    "quantity": quantity
                })
            else:
                limit_price = current_price * (1.001 if action == "buy" else 0.999)
                order_result = await self.alpaca_mcp._place_limit_order({
                    "symbol": symbol,
                    "side": action,
                    "quantity": quantity,
                    "limit_price": limit_price
                })
            
            return {
                "success": True,
                "order_result": order_result[0].text if order_result else "Failed",
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": current_price
            }
            
        except Exception as e:
            logger.error(f"❌ Error executing manual trade: {e}")
            return {"error": str(e)}
    
    async def run_backtest(self, symbols: List[str], days: int = 7) -> Dict[str, Any]:
        """Run backtest on historical data"""
        try:
            logger.info(f"🔄 Running backtest for {symbols} over {days} days")
            
            backtest_results = {
                "symbols": symbols,
                "days": days,
                "trades": [],
                "performance": {}
            }
            
            for symbol in symbols:
                # Get historical data
                historical_data = await self.data_streamer.get_historical_data(
                    symbol, "1Hour", days * 24
                )
                
                if not historical_data:
                    continue
                
                # Simulate trading on historical data
                symbol_trades = []
                for i in range(20, len(historical_data)):  # Start after 20 bars for indicators
                    # Create market context
                    market_context = {
                        "symbol": symbol,
                        "bars": historical_data[i-20:i],
                        "latest_price": historical_data[i]["close"],
                        "technical_indicators": self.data_streamer._calculate_technical_indicators(
                            historical_data[i-20:i]
                        )
                    }
                    
                    # Generate signal
                    signal = self.model.generate_signal(market_context)
                    
                    if signal["action"] != "hold":
                        symbol_trades.append({
                            "timestamp": historical_data[i]["timestamp"],
                            "signal": signal,
                            "price": historical_data[i]["close"]
                        })
                
                backtest_results["trades"].extend(symbol_trades)
            
            # Calculate performance metrics
            backtest_results["performance"] = self._calculate_backtest_performance(
                backtest_results["trades"]
            )
            
            logger.info(f"✅ Backtest completed: {len(backtest_results['trades'])} trades")
            return backtest_results
            
        except Exception as e:
            logger.error(f"❌ Error running backtest: {e}")
            return {"error": str(e)}
    
    def _calculate_backtest_performance(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate backtest performance metrics"""
        try:
            if not trades:
                return {"error": "No trades to analyze"}
            
            total_trades = len(trades)
            
            # Calculate actual P&L based on price movements
            total_pnl = 0.0
            winning_trades = 0
            losing_trades = 0
            pnl_list = []
            
            for i, trade in enumerate(trades):
                signal = trade["signal"]
                entry_price = trade["price"]
                
                # For backtesting, we need to simulate exit prices
                # This is a simplified approach - in practice, you'd use actual historical data
                if i < len(trades) - 1:
                    # Use next trade price as exit price (simplified)
                    exit_price = trades[i + 1]["price"]
                else:
                    # For last trade, use a small random movement
                    import random
                    movement = random.uniform(-0.02, 0.02)  # ±2% movement
                    exit_price = entry_price * (1 + movement)
                
                # Calculate P&L based on signal action
                if signal["action"] == "buy":
                    pnl = (exit_price - entry_price) / entry_price
                elif signal["action"] == "sell":
                    pnl = (entry_price - exit_price) / entry_price
                else:
                    pnl = 0.0
                
                # Apply confidence as a multiplier
                pnl *= signal["confidence"]
                
                total_pnl += pnl
                pnl_list.append(pnl)
                
                if pnl > 0:
                    winning_trades += 1
                elif pnl < 0:
                    losing_trades += 1
            
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Calculate additional metrics
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
            max_pnl = max(pnl_list) if pnl_list else 0
            min_pnl = min(pnl_list) if pnl_list else 0
            
            # Calculate Sharpe ratio (simplified)
            if len(pnl_list) > 1:
                import statistics
                pnl_std = statistics.stdev(pnl_list)
                sharpe_ratio = (avg_pnl / pnl_std) if pnl_std > 0 else 0
            else:
                sharpe_ratio = 0
            
            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": round(win_rate, 2),
                "total_pnl": round(total_pnl, 4),
                "avg_pnl": round(avg_pnl, 4),
                "max_pnl": round(max_pnl, 4),
                "min_pnl": round(min_pnl, 4),
                "sharpe_ratio": round(sharpe_ratio, 4),
                "avg_confidence": round(
                    sum(trade["signal"]["confidence"] for trade in trades) / total_trades, 2
                ) if total_trades > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating backtest performance: {e}")
            return {"error": str(e)}


async def main():
    """Main function to run the trading agent"""
    # Initialize and start the agent
    agent = CryptoTradingAgent()
    
    try:
        await agent.initialize()
        
        # Start trading
        await agent.start_trading()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await agent.stop_trading()


if __name__ == "__main__":
    asyncio.run(main())
