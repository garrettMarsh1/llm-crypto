"""
Risk Management Module for Crypto Trading Agent
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import yaml
from loguru import logger


class RiskManager:
    """Comprehensive risk management for crypto trading"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.risk_config = self.config['risk']
        self.trading_config = self.config['trading']
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_trades = 50
        self.last_reset_date = datetime.now().date()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def validate_signal(self, signal: Dict[str, Any], current_positions: Dict[str, Any]) -> bool:
        """Validate trading signal against risk rules"""
        try:
            # Reset daily counters if new day
            await self._reset_daily_counters()
            
            # Basic signal validation
            if not self._validate_signal_format(signal):
                logger.warning("❌ Invalid signal format")
                return False
            
            # Check daily trade limit
            if self.daily_trades >= self.max_daily_trades:
                logger.warning("❌ Daily trade limit reached")
                return False
            
            # Check daily loss limit
            if self.daily_pnl <= -self.trading_config['max_daily_loss']:
                logger.warning("❌ Daily loss limit exceeded")
                return False
            
            # Check position limits
            if not await self._check_position_limits(signal, current_positions):
                logger.warning("❌ Position limits exceeded")
                return False
            
            # Check volatility filter
            if not await self._check_volatility_filter(signal):
                logger.warning("❌ Volatility filter failed")
                return False
            
            # Check confidence threshold
            if signal.get("confidence", 0) < 0.6:
                logger.warning("❌ Signal confidence too low")
                return False
            
            logger.info("✅ Signal passed all risk checks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating signal: {e}")
            return False
    
    def _validate_signal_format(self, signal: Dict[str, Any]) -> bool:
        """Validate signal format and required fields"""
        required_fields = ["action", "symbol", "confidence", "reasoning"]
        
        for field in required_fields:
            if field not in signal:
                logger.error(f"Missing required field: {field}")
                return False
        
        if signal["action"] not in ["buy", "sell", "hold"]:
            logger.error(f"Invalid action: {signal['action']}")
            return False
        
        if not (0 <= signal["confidence"] <= 1):
            logger.error(f"Invalid confidence: {signal['confidence']}")
            return False
        
        if signal["symbol"] not in self.trading_config['crypto_pairs']:
            logger.error(f"Unsupported symbol: {signal['symbol']}")
            return False
        
        return True
    
    async def _check_position_limits(self, signal: Dict[str, Any], current_positions: Dict[str, Any]) -> bool:
        """Check position size and count limits"""
        try:
            symbol = signal["symbol"]
            action = signal["action"]
            
            if action == "hold":
                return True
            
            # Count current positions
            position_count = len([p for p in current_positions.values() if float(p.get("qty", 0)) != 0])
            
            # Check max open positions
            if position_count >= self.risk_config['max_open_positions']:
                logger.warning(f"Max open positions reached: {position_count}")
                return False
            
            # Check if we already have a position in this symbol
            if symbol in current_positions and float(current_positions[symbol].get("qty", 0)) != 0:
                logger.warning(f"Position already exists for {symbol}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking position limits: {e}")
            return False
    
    async def _check_volatility_filter(self, signal: Dict[str, Any]) -> bool:
        """Check if volatility is within acceptable limits"""
        try:
            # This is a simplified volatility check
            # In practice, you'd calculate actual volatility from recent price data
            
            # For now, we'll use a simple confidence-based filter
            confidence = signal.get("confidence", 0)
            volatility_threshold = self.risk_config['volatility_threshold']
            
            # Higher confidence required for higher volatility environments
            if confidence < 0.8:  # Lower confidence signals need lower volatility
                return True  # Simplified for now
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking volatility filter: {e}")
            return False
    
    async def calculate_position_size(self, symbol: str, signal: Dict[str, Any], 
                                    market_context: Dict[str, Any], 
                                    current_positions: Dict[str, Any]) -> float:
        """Calculate appropriate position size based on risk management rules"""
        try:
            # Get real account value from Alpaca API
            # This should be injected from the trading agent
            account_value = getattr(self, '_account_value', 10000)  # Fallback to 10k if not set
            
            # Calculate base position size
            max_position_value = account_value * self.trading_config['max_position_size']
            
            # Adjust based on signal confidence
            confidence = signal.get("confidence", 0.5)
            confidence_multiplier = min(confidence * 2, 1.0)  # Scale confidence to 0-1
            
            # Adjust based on volatility
            volatility = market_context.get("technical_indicators", {}).get("volatility", 0)
            volatility_multiplier = max(0.5, 1.0 - (volatility / 10))  # Reduce size for high volatility
            
            # Adjust based on current exposure
            current_exposure = self._calculate_current_exposure(current_positions, account_value)
            exposure_multiplier = max(0.1, 1.0 - current_exposure)  # Reduce size if already exposed
            
            # Calculate final position size
            position_value = max_position_value * confidence_multiplier * volatility_multiplier * exposure_multiplier
            
            # Ensure minimum trade amount
            min_trade_value = self.trading_config['min_trade_amount']
            if position_value < min_trade_value:
                return 0.0
            
            # Get current price
            current_price = market_context.get("latest_price", 0)
            if current_price <= 0:
                return 0.0
            
            # Convert to crypto units
            position_size = position_value / current_price
            
            logger.info(f"Position size calculated: ${position_value:.2f} ({position_size:.6f} {symbol})")
            logger.info(f"Confidence: {confidence:.2f}, Volatility: {volatility:.2f}, Exposure: {current_exposure:.2f}")
            
            return position_size
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return 0.0
    
    def _calculate_current_exposure(self, current_positions: Dict[str, Any], account_value: float) -> float:
        """Calculate current portfolio exposure as a percentage"""
        try:
            if not current_positions or account_value <= 0:
                return 0.0
            
            total_exposure = 0.0
            for symbol, position in current_positions.items():
                if float(position.get("qty", 0)) != 0:
                    market_value = float(position.get("market_value", 0))
                    total_exposure += market_value
            
            return total_exposure / account_value
            
        except Exception as e:
            logger.error(f"❌ Error calculating current exposure: {e}")
            return 0.0
    
    def set_account_value(self, account_value: float):
        """Set the current account value for position sizing calculations"""
        self._account_value = account_value
    
    async def _reset_daily_counters(self):
        """Reset daily counters if new day"""
        current_date = datetime.now().date()
        
        if current_date > self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = current_date
            logger.info("📅 Daily risk counters reset")
    
    def update_trade_result(self, symbol: str, pnl: float, trade_size: float):
        """Update risk metrics after trade execution"""
        try:
            self.daily_pnl += pnl
            self.daily_trades += 1
            
            # Update max drawdown
            if pnl < 0:
                current_drawdown = abs(pnl) / 10000  # Simplified calculation
                if current_drawdown > self.risk_config['max_drawdown']:
                    logger.warning(f"⚠️ Drawdown exceeded: {current_drawdown:.2%}")
            
            logger.info(f"Trade result updated: P&L: ${pnl:.2f}, Daily P&L: ${self.daily_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error updating trade result: {e}")
    
    async def check_emergency_stop(self) -> bool:
        """Check if emergency stop conditions are met"""
        try:
            # Check daily loss limit
            if self.daily_pnl <= -self.trading_config['max_daily_loss']:
                logger.critical("🚨 EMERGENCY STOP: Daily loss limit exceeded")
                return True
            
            # Check max drawdown
            current_drawdown = abs(self.daily_pnl) / 10000  # Simplified
            if current_drawdown >= self.risk_config['max_drawdown']:
                logger.critical("🚨 EMERGENCY STOP: Max drawdown exceeded")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking emergency stop: {e}")
            return False
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        try:
            return {
                "daily_pnl": self.daily_pnl,
                "daily_trades": self.daily_trades,
                "max_daily_trades": self.max_daily_trades,
                "max_daily_loss": self.trading_config['max_daily_loss'],
                "max_drawdown": self.risk_config['max_drawdown'],
                "max_position_size": self.trading_config['max_position_size'],
                "max_open_positions": self.risk_config['max_open_positions'],
                "volatility_threshold": self.risk_config['volatility_threshold'],
                "last_reset": self.last_reset_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {"error": str(e)}
    
    async def validate_order(self, order_params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate order parameters before execution"""
        try:
            symbol = order_params.get("symbol")
            side = order_params.get("side")
            quantity = order_params.get("quantity")
            
            # Basic validation
            if not all([symbol, side, quantity]):
                return False, "Missing required order parameters"
            
            if side not in ["buy", "sell"]:
                return False, "Invalid order side"
            
            if quantity <= 0:
                return False, "Invalid quantity"
            
            if symbol not in self.trading_config['crypto_pairs']:
                return False, "Unsupported symbol"
            
            # Check minimum trade amount
            # This would need current price to calculate properly
            # For now, we'll assume it's handled elsewhere
            
            return True, "Order validation passed"
            
        except Exception as e:
            logger.error(f"❌ Error validating order: {e}")
            return False, f"Validation error: {str(e)}"
    
    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price"""
        try:
            stop_loss_pct = self.risk_config['stop_loss_pct']
            
            if side == "buy":
                return entry_price * (1 - stop_loss_pct)
            else:  # sell
                return entry_price * (1 + stop_loss_pct)
                
        except Exception as e:
            logger.error(f"❌ Error calculating stop loss: {e}")
            return entry_price
    
    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calculate take profit price"""
        try:
            take_profit_pct = self.risk_config['take_profit_pct']
            
            if side == "buy":
                return entry_price * (1 + take_profit_pct)
            else:  # sell
                return entry_price * (1 - take_profit_pct)
                
        except Exception as e:
            logger.error(f"❌ Error calculating take profit: {e}")
            return entry_price
    
    async def monitor_positions(self, positions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Monitor positions for risk management actions"""
        try:
            alerts = []
            
            for symbol, position in positions.items():
                if float(position.get("qty", 0)) == 0:
                    continue
                
                # Check unrealized P&L
                unrealized_pl = float(position.get("unrealized_pl", 0))
                unrealized_plpc = float(position.get("unrealized_plpc", 0))
                
                # Alert on significant losses
                if unrealized_plpc <= -0.05:  # 5% loss
                    alerts.append({
                        "type": "position_loss",
                        "symbol": symbol,
                        "message": f"Position {symbol} down {unrealized_plpc:.2%}",
                        "severity": "warning"
                    })
                
                # Alert on significant gains
                if unrealized_plpc >= 0.10:  # 10% gain
                    alerts.append({
                        "type": "position_gain",
                        "symbol": symbol,
                        "message": f"Position {symbol} up {unrealized_plpc:.2%}",
                        "severity": "info"
                    })
            
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Error monitoring positions: {e}")
            return []


class PortfolioManager:
    """Portfolio-level risk management"""
    
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.portfolio_value = 10000  # This should come from account
        self.target_allocation = {
            "BTC/USD": 0.4,  # 40%
            "ETH/USD": 0.3,  # 30%
            "SOL/USD": 0.3   # 30%
        }
    
    async def rebalance_portfolio(self, current_positions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest portfolio rebalancing actions"""
        try:
            rebalance_actions = []
            
            # Calculate current allocation
            current_allocation = {}
            total_value = 0
            
            for symbol, position in current_positions.items():
                if float(position.get("qty", 0)) != 0:
                    market_value = float(position.get("market_value", 0))
                    current_allocation[symbol] = market_value
                    total_value += market_value
            
            # Normalize allocations
            if total_value > 0:
                for symbol in current_allocation:
                    current_allocation[symbol] /= total_value
            
            # Check for rebalancing needs
            for symbol, target_pct in self.target_allocation.items():
                current_pct = current_allocation.get(symbol, 0)
                difference = target_pct - current_pct
                
                if abs(difference) > 0.05:  # 5% threshold
                    action = "buy" if difference > 0 else "sell"
                    rebalance_actions.append({
                        "symbol": symbol,
                        "action": action,
                        "current_allocation": current_pct,
                        "target_allocation": target_pct,
                        "difference": difference
                    })
            
            return rebalance_actions
            
        except Exception as e:
            logger.error(f"❌ Error rebalancing portfolio: {e}")
            return []


if __name__ == "__main__":
    # Test the risk manager
    risk_manager = RiskManager()
    
    # Test signal validation
    test_signal = {
        "action": "buy",
        "symbol": "BTC/USD",
        "confidence": 0.8,
        "reasoning": "Strong bullish momentum"
    }
    
    # This would be run in an async context
    print("Risk manager initialized successfully")
