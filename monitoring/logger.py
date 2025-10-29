"""
Advanced Logging and Monitoring System for Crypto Trading Agent
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
import yaml

from loguru import logger
import pandas as pd


class TradingLogger:
    """Advanced logging system for trading operations"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.log_config = self.config['logging']
        self.setup_logging()
        self.trade_logs = []
        self.performance_logs = []
        self.error_logs = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup loguru logging configuration"""
        try:
            # Remove default handler
            logger.remove()
            
            # Create logs directory
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # Console logging
            logger.add(
                sys.stderr,
                level=self.log_config['level'],
                format=self.log_config['format'],
                colorize=True
            )
            
            # File logging with rotation
            logger.add(
                self.log_config['file'],
                level=self.log_config['level'],
                format=self.log_config['format'],
                rotation=self.log_config['rotation'],
                retention=self.log_config['retention'],
                compression="zip"
            )
            
            # Separate trade log file
            logger.add(
                "logs/trades_{time}.log",
                level="INFO",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                rotation="1 day",
                retention="90 days",
                filter=lambda record: "TRADE" in record["message"]
            )
            
            # Separate error log file
            logger.add(
                "logs/errors_{time}.log",
                level="ERROR",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
                rotation="1 day",
                retention="30 days"
            )
            
            # Performance metrics log
            logger.add(
                "logs/performance_{time}.log",
                level="INFO",
                format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
                rotation="1 day",
                retention="180 days",
                filter=lambda record: "PERF" in record["message"]
            )
            
            logger.info("✅ Logging system initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to setup logging: {e}")
            raise
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade execution details"""
        try:
            trade_log = {
                "timestamp": datetime.now().isoformat(),
                "type": "trade_execution",
                "data": trade_data
            }
            
            self.trade_logs.append(trade_log)
            
            logger.info(f"TRADE | {json.dumps(trade_data, indent=2)}")
            
            # Save to JSONL file
            self._save_to_jsonl("logs/trades.jsonl", trade_log)
            
        except Exception as e:
            logger.error(f"❌ Error logging trade: {e}")
    
    def log_signal(self, signal_data: Dict[str, Any]):
        """Log trading signal generation"""
        try:
            signal_log = {
                "timestamp": datetime.now().isoformat(),
                "type": "signal_generation",
                "data": signal_data
            }
            
            logger.info(f"SIGNAL | {json.dumps(signal_data, indent=2)}")
            
            # Save to JSONL file
            self._save_to_jsonl("logs/signals.jsonl", signal_log)
            
        except Exception as e:
            logger.error(f"❌ Error logging signal: {e}")
    
    def log_performance(self, performance_data: Dict[str, Any]):
        """Log performance metrics"""
        try:
            perf_log = {
                "timestamp": datetime.now().isoformat(),
                "type": "performance_update",
                "data": performance_data
            }
            
            self.performance_logs.append(perf_log)
            
            logger.info(f"PERF | {json.dumps(performance_data, indent=2)}")
            
            # Save to JSONL file
            self._save_to_jsonl("logs/performance.jsonl", perf_log)
            
        except Exception as e:
            logger.error(f"❌ Error logging performance: {e}")
    
    def log_error(self, error_data: Dict[str, Any]):
        """Log error details"""
        try:
            error_log = {
                "timestamp": datetime.now().isoformat(),
                "type": "error",
                "data": error_data
            }
            
            self.error_logs.append(error_log)
            
            logger.error(f"ERROR | {json.dumps(error_data, indent=2)}")
            
            # Save to JSONL file
            self._save_to_jsonl("logs/errors.jsonl", error_log)
            
        except Exception as e:
            print(f"❌ Error logging error: {e}")
    
    def log_risk_event(self, risk_data: Dict[str, Any]):
        """Log risk management events"""
        try:
            risk_log = {
                "timestamp": datetime.now().isoformat(),
                "type": "risk_event",
                "data": risk_data
            }
            
            logger.warning(f"RISK | {json.dumps(risk_data, indent=2)}")
            
            # Save to JSONL file
            self._save_to_jsonl("logs/risk_events.jsonl", risk_log)
            
        except Exception as e:
            logger.error(f"❌ Error logging risk event: {e}")
    
    def _save_to_jsonl(self, filepath: str, data: Dict[str, Any]):
        """Save data to JSONL file"""
        try:
            with open(filepath, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.error(f"❌ Error saving to JSONL: {e}")
    
    def get_trade_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get trade summary for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_trades = [
                trade for trade in self.trade_logs
                if datetime.fromisoformat(trade["timestamp"]) > cutoff_time
            ]
            
            if not recent_trades:
                return {"message": "No trades in specified period"}
            
            # Calculate summary statistics
            total_trades = len(recent_trades)
            symbols = {}
            actions = {"buy": 0, "sell": 0, "hold": 0}
            
            for trade in recent_trades:
                trade_data = trade["data"]
                symbol = trade_data.get("symbol", "unknown")
                action = trade_data.get("action", "unknown")
                
                if symbol not in symbols:
                    symbols[symbol] = 0
                symbols[symbol] += 1
                
                if action in actions:
                    actions[action] += 1
            
            return {
                "period_hours": hours,
                "total_trades": total_trades,
                "symbols": symbols,
                "actions": actions,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting trade summary: {e}")
            return {"error": str(e)}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from logs"""
        try:
            if not self.performance_logs:
                return {"message": "No performance data available"}
            
            # Get latest performance data
            latest_perf = self.performance_logs[-1]["data"]
            
            return {
                "latest_performance": latest_perf,
                "total_logs": len(self.performance_logs),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance summary: {e}")
            return {"error": str(e)}


class PerformanceMonitor:
    """Performance monitoring and alerting system"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.monitoring_config = self.config['monitoring']
        self.alerts = []
        self.metrics_history = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def check_performance_alerts(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for performance alerts"""
        try:
            alerts = []
            
            # Check drawdown alert
            drawdown = current_metrics.get("max_drawdown", 0)
            if drawdown >= self.monitoring_config['alert_drawdown_pct']:
                alerts.append({
                    "type": "drawdown_alert",
                    "severity": "high",
                    "message": f"Drawdown exceeded threshold: {drawdown:.2%}",
                    "value": drawdown,
                    "threshold": self.monitoring_config['alert_drawdown_pct']
                })
            
            # Check win rate
            win_rate = current_metrics.get("win_rate", 0)
            if win_rate < 0.3:  # Less than 30% win rate
                alerts.append({
                    "type": "low_win_rate",
                    "severity": "medium",
                    "message": f"Low win rate: {win_rate:.2%}",
                    "value": win_rate,
                    "threshold": 0.3
                })
            
            # Check trade frequency
            trades_per_hour = current_metrics.get("trades_per_hour", 0)
            if trades_per_hour > 10:  # More than 10 trades per hour
                alerts.append({
                    "type": "high_frequency",
                    "severity": "medium",
                    "message": f"High trading frequency: {trades_per_hour:.1f} trades/hour",
                    "value": trades_per_hour,
                    "threshold": 10
                })
            
            # Store alerts
            self.alerts.extend(alerts)
            
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Error checking performance alerts: {e}")
            return []
    
    def calculate_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics from trade data"""
        try:
            if not trades:
                return {"error": "No trades to analyze"}
            
            # Basic metrics
            total_trades = len(trades)
            winning_trades = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
            losing_trades = total_trades - winning_trades
            
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # P&L metrics
            total_pnl = sum(trade.get("pnl", 0) for trade in trades)
            avg_win = sum(trade.get("pnl", 0) for trade in trades if trade.get("pnl", 0) > 0) / max(winning_trades, 1)
            avg_loss = sum(trade.get("pnl", 0) for trade in trades if trade.get("pnl", 0) < 0) / max(losing_trades, 1)
            
            # Drawdown calculation
            cumulative_pnl = 0
            peak = 0
            max_drawdown = 0
            
            for trade in trades:
                cumulative_pnl += trade.get("pnl", 0)
                if cumulative_pnl > peak:
                    peak = cumulative_pnl
                drawdown = peak - cumulative_pnl
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            max_drawdown_pct = (max_drawdown / max(peak, 1)) * 100 if peak > 0 else 0
            
            # Time-based metrics
            if trades:
                first_trade = min(trade.get("timestamp", "") for trade in trades)
                last_trade = max(trade.get("timestamp", "") for trade in trades)
                
                try:
                    start_time = datetime.fromisoformat(first_trade.replace('Z', '+00:00'))
                    end_time = datetime.fromisoformat(last_trade.replace('Z', '+00:00'))
                    duration_hours = (end_time - start_time).total_seconds() / 3600
                    trades_per_hour = total_trades / max(duration_hours, 1)
                except:
                    trades_per_hour = 0
            else:
                trades_per_hour = 0
            
            metrics = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": round(win_rate, 2),
                "total_pnl": round(total_pnl, 2),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "max_drawdown": round(max_drawdown, 2),
                "max_drawdown_pct": round(max_drawdown_pct, 2),
                "trades_per_hour": round(trades_per_hour, 2),
                "profit_factor": round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0
            }
            
            # Store in history
            self.metrics_history.append({
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating metrics: {e}")
            return {"error": str(e)}
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts"""
        try:
            recent_alerts = self.alerts[-50:]  # Last 50 alerts
            
            alert_counts = {}
            for alert in recent_alerts:
                alert_type = alert["type"]
                alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
            
            return {
                "total_alerts": len(self.alerts),
                "recent_alerts": len(recent_alerts),
                "alert_counts": alert_counts,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting alert summary: {e}")
            return {"error": str(e)}


class HealthChecker:
    """System health monitoring"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.health_checks = []
        self.last_check = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def check_system_health(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Check overall system health"""
        try:
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "healthy",
                "components": {},
                "issues": []
            }
            
            # Check each component
            for component_name, component in components.items():
                component_health = await self._check_component_health(component_name, component)
                health_status["components"][component_name] = component_health
                
                if component_health["status"] != "healthy":
                    health_status["overall_status"] = "degraded"
                    health_status["issues"].append(f"{component_name}: {component_health['message']}")
            
            # Store health check
            self.health_checks.append(health_status)
            self.last_check = datetime.now()
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Error checking system health: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "error",
                "error": str(e)
            }
    
    async def _check_component_health(self, name: str, component: Any) -> Dict[str, Any]:
        """Check health of individual component"""
        try:
            if name == "model":
                return await self._check_model_health(component)
            elif name == "data_streamer":
                return await self._check_data_streamer_health(component)
            elif name == "risk_manager":
                return await self._check_risk_manager_health(component)
            elif name == "alpaca_mcp":
                return await self._check_alpaca_mcp_health(component)
            else:
                return {
                    "status": "unknown",
                    "message": f"Unknown component: {name}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Health check failed: {str(e)}"
            }
    
    async def _check_model_health(self, model) -> Dict[str, Any]:
        """Check model health"""
        try:
            if model is None:
                return {"status": "unhealthy", "message": "Model not initialized"}
            
            # Check if model can generate
            test_context = {"symbol": "BTC/USD", "bars": [], "positions": []}
            signal = model.generate_signal(test_context)
            
            if signal and "action" in signal:
                return {"status": "healthy", "message": "Model responding correctly"}
            else:
                return {"status": "degraded", "message": "Model not generating valid signals"}
                
        except Exception as e:
            return {"status": "unhealthy", "message": f"Model error: {str(e)}"}
    
    async def _check_data_streamer_health(self, streamer) -> Dict[str, Any]:
        """Check data streamer health"""
        try:
            if streamer is None:
                return {"status": "unhealthy", "message": "Data streamer not initialized"}
            
            if not streamer.running:
                return {"status": "degraded", "message": "Data streamer not running"}
            
            return {"status": "healthy", "message": "Data streamer running"}
            
        except Exception as e:
            return {"status": "unhealthy", "message": f"Data streamer error: {str(e)}"}
    
    async def _check_risk_manager_health(self, risk_manager) -> Dict[str, Any]:
        """Check risk manager health"""
        try:
            if risk_manager is None:
                return {"status": "unhealthy", "message": "Risk manager not initialized"}
            
            # Check if emergency stop is triggered
            emergency_stop = await risk_manager.check_emergency_stop()
            if emergency_stop:
                return {"status": "unhealthy", "message": "Emergency stop triggered"}
            
            return {"status": "healthy", "message": "Risk manager operational"}
            
        except Exception as e:
            return {"status": "unhealthy", "message": f"Risk manager error: {str(e)}"}
    
    async def _check_alpaca_mcp_health(self, alpaca_mcp) -> Dict[str, Any]:
        """Check Alpaca MCP health"""
        try:
            if alpaca_mcp is None:
                return {"status": "unhealthy", "message": "Alpaca MCP not initialized"}
            
            # Try to get account info
            account_info = await alpaca_mcp._get_account_info()
            if account_info and len(account_info) > 0:
                return {"status": "healthy", "message": "Alpaca MCP responding"}
            else:
                return {"status": "degraded", "message": "Alpaca MCP not responding"}
                
        except Exception as e:
            return {"status": "unhealthy", "message": f"Alpaca MCP error: {str(e)}"}


if __name__ == "__main__":
    # Test the logging system
    trading_logger = TradingLogger()
    
    # Test trade logging
    test_trade = {
        "symbol": "BTC/USD",
        "action": "buy",
        "quantity": 0.001,
        "price": 45000,
        "confidence": 0.8
    }
    
    trading_logger.log_trade(test_trade)
    print("Logging system test completed")
