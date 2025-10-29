"""
Real-time monitoring dashboard for the trading agent
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.trading_agent import CryptoTradingAgent
from monitoring.logger import TradingLogger, PerformanceMonitor, HealthChecker
from loguru import logger


class TradingDashboard:
    """Real-time trading dashboard"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.agent = None
        self.trading_logger = TradingLogger(config_path)
        self.performance_monitor = PerformanceMonitor(config_path)
        self.health_checker = HealthChecker(config_path)
        self.running = False
        
    async def initialize(self):
        """Initialize dashboard components"""
        try:
            self.agent = CryptoTradingAgent(self.config_path)
            await self.agent.initialize()
            logger.info("✅ Dashboard initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize dashboard: {e}")
            raise
    
    async def start_monitoring(self, refresh_interval: int = 5):
        """Start monitoring loop"""
        self.running = True
        logger.info(f"📊 Starting monitoring dashboard (refresh every {refresh_interval}s)")
        
        try:
            while self.running:
                await self._update_dashboard()
                await asyncio.sleep(refresh_interval)
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")
        finally:
            self.running = False
    
    async def _update_dashboard(self):
        """Update dashboard display"""
        try:
            # Clear screen (works on most terminals)
            print("\033[2J\033[H", end="")
            
            # Header
            print("=" * 80)
            print(f"🚀 CRYPTO TRADING AGENT DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            
            # Account Status
            await self._display_account_status()
            
            # Performance Metrics
            await self._display_performance_metrics()
            
            # Recent Trades
            await self._display_recent_trades()
            
            # System Health
            await self._display_system_health()
            
            # Alerts
            await self._display_alerts()
            
            print("=" * 80)
            print("Press Ctrl+C to stop monitoring")
            
        except Exception as e:
            logger.error(f"❌ Error updating dashboard: {e}")
    
    async def _display_account_status(self):
        """Display account status"""
        try:
            if not self.agent:
                print("❌ Agent not initialized")
                return
            
            status = await self.agent.get_account_status()
            
            print("\n💰 ACCOUNT STATUS")
            print("-" * 40)
            
            if "account" in status and status["account"]:
                account = status["account"]
                print(f"Equity: ${account.get('equity', 0):,.2f}")
                print(f"Cash: ${account.get('cash', 0):,.2f}")
                print(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
                print(f"Trading Blocked: {account.get('trading_blocked', 'Unknown')}")
            
            if "positions" in status and status["positions"]:
                print(f"\nOpen Positions: {len(status['positions'])}")
                for pos in status["positions"][:3]:  # Show first 3
                    print(f"  {pos['symbol']}: {pos['qty']} @ ${pos['market_value']:,.2f} (P&L: ${pos['unrealized_pl']:,.2f})")
            else:
                print("No open positions")
                
        except Exception as e:
            print(f"❌ Error displaying account status: {e}")
    
    async def _display_performance_metrics(self):
        """Display performance metrics"""
        try:
            print("\n📈 PERFORMANCE METRICS")
            print("-" * 40)
            
            # Get trade summary
            trade_summary = self.trading_logger.get_trade_summary(24)
            
            if "total_trades" in trade_summary:
                print(f"Trades (24h): {trade_summary['total_trades']}")
                print(f"Symbols: {trade_summary.get('symbols', {})}")
                print(f"Actions: {trade_summary.get('actions', {})}")
            
            # Get performance summary
            perf_summary = self.trading_logger.get_performance_summary()
            if "latest_performance" in perf_summary:
                perf = perf_summary["latest_performance"]
                print(f"Win Rate: {perf.get('win_rate', 0):.1f}%")
                print(f"Total P&L: ${perf.get('total_pnl', 0):,.2f}")
                print(f"Max Drawdown: {perf.get('max_drawdown_pct', 0):.1f}%")
            
        except Exception as e:
            print(f"❌ Error displaying performance metrics: {e}")
    
    async def _display_recent_trades(self):
        """Display recent trades"""
        try:
            print("\n📊 RECENT TRADES")
            print("-" * 40)
            
            # Read recent trades from log file
            trade_file = Path("logs/trades.jsonl")
            if trade_file.exists():
                trades = []
                with open(trade_file, "r") as f:
                    for line in f:
                        try:
                            trade = json.loads(line.strip())
                            if trade.get("type") == "trade_execution":
                                trades.append(trade)
                        except:
                            continue
                
                # Show last 5 trades
                for trade in trades[-5:]:
                    data = trade.get("data", {})
                    timestamp = trade.get("timestamp", "")[:19]  # Remove microseconds
                    print(f"{timestamp} | {data.get('symbol', 'N/A')} | {data.get('action', 'N/A')} | ${data.get('price', 0):,.2f}")
            else:
                print("No trades found")
                
        except Exception as e:
            print(f"❌ Error displaying recent trades: {e}")
    
    async def _display_system_health(self):
        """Display system health"""
        try:
            print("\n🏥 SYSTEM HEALTH")
            print("-" * 40)
            
            if self.agent:
                components = {
                    "model": self.agent.model,
                    "data_streamer": self.agent.data_streamer,
                    "risk_manager": self.agent.risk_manager,
                    "alpaca_mcp": self.agent.alpaca_mcp
                }
                
                health = await self.health_checker.check_system_health(components)
                
                print(f"Overall Status: {health.get('overall_status', 'Unknown')}")
                
                for component, status in health.get("components", {}).items():
                    status_icon = "✅" if status["status"] == "healthy" else "⚠️" if status["status"] == "degraded" else "❌"
                    print(f"  {status_icon} {component}: {status['message']}")
                
                if health.get("issues"):
                    print("\nIssues:")
                    for issue in health["issues"]:
                        print(f"  ⚠️ {issue}")
            else:
                print("❌ Agent not initialized")
                
        except Exception as e:
            print(f"❌ Error displaying system health: {e}")
    
    async def _display_alerts(self):
        """Display alerts"""
        try:
            print("\n🚨 ALERTS")
            print("-" * 40)
            
            alert_summary = self.performance_monitor.get_alert_summary()
            
            if alert_summary.get("recent_alerts", 0) > 0:
                print(f"Recent Alerts: {alert_summary['recent_alerts']}")
                print(f"Alert Types: {alert_summary.get('alert_counts', {})}")
            else:
                print("No recent alerts")
                
        except Exception as e:
            print(f"❌ Error displaying alerts: {e}")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Trading Agent Dashboard")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--refresh", type=int, default=5, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    dashboard = TradingDashboard(args.config)
    
    try:
        await dashboard.initialize()
        await dashboard.start_monitoring(args.refresh)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
    finally:
        dashboard.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
