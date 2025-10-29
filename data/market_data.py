"""
Real-time Market Data Pipeline for Crypto Trading
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import yaml

from alpaca.data.live import CryptoDataStream
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from loguru import logger


class MarketDataBuffer:
    """Buffer for storing recent market data"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.bars = deque(maxlen=max_size)
        self.quotes = deque(maxlen=max_size)
        self.trades = deque(maxlen=max_size)
        self.lock = asyncio.Lock()
    
    async def add_bar(self, bar_data: Dict[str, Any]):
        """Add bar data to buffer"""
        async with self.lock:
            self.bars.append({
                "timestamp": bar_data.timestamp.isoformat(),
                "symbol": bar_data.symbol,
                "open": float(bar_data.open),
                "high": float(bar_data.high),
                "low": float(bar_data.low),
                "close": float(bar_data.close),
                "volume": float(bar_data.volume),
                "trade_count": int(bar_data.trade_count),
                "vwap": float(bar_data.vwap)
            })
    
    async def add_quote(self, quote_data: Dict[str, Any]):
        """Add quote data to buffer"""
        async with self.lock:
            self.quotes.append({
                "timestamp": quote_data.timestamp.isoformat(),
                "symbol": quote_data.symbol,
                "bid_price": float(quote_data.bid_price),
                "ask_price": float(quote_data.ask_price),
                "bid_size": float(quote_data.bid_size),
                "ask_size": float(quote_data.ask_size)
            })
    
    async def add_trade(self, trade_data: Dict[str, Any]):
        """Add trade data to buffer"""
        async with self.lock:
            self.trades.append({
                "timestamp": trade_data.timestamp.isoformat(),
                "symbol": trade_data.symbol,
                "price": float(trade_data.price),
                "size": float(trade_data.size),
                "trade_id": trade_data.trade_id
            })
    
    async def get_recent_bars(self, symbol: str, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent bars for a symbol"""
        async with self.lock:
            symbol_bars = [bar for bar in self.bars if bar["symbol"] == symbol]
            return symbol_bars[-count:] if symbol_bars else []
    
    async def get_recent_quotes(self, symbol: str, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent quotes for a symbol"""
        async with self.lock:
            symbol_quotes = [quote for quote in self.quotes if quote["symbol"] == symbol]
            return symbol_quotes[-count:] if symbol_quotes else []
    
    async def get_recent_trades(self, symbol: str, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent trades for a symbol"""
        async with self.lock:
            symbol_trades = [trade for trade in self.trades if trade["symbol"] == symbol]
            return symbol_trades[-count:] if symbol_trades else []
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        async with self.lock:
            symbol_bars = [bar for bar in self.bars if bar["symbol"] == symbol]
            if symbol_bars:
                return symbol_bars[-1]["close"]
            return None


class MarketDataStreamer:
    """Real-time market data streaming from Alpaca"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.data_client = None
        self.stream_client = None
        self.buffer = MarketDataBuffer()
        self.callbacks = []
        self.running = False
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def initialize(self):
        """Initialize Alpaca data clients"""
        try:
            self.data_client = CryptoHistoricalDataClient(
                api_key=self.config['alpaca']['api_key'],
                secret_key=self.config['alpaca']['api_secret']
            )
            
            self.stream_client = CryptoDataStream(
                api_key=self.config['alpaca']['api_key'],
                secret_key=self.config['alpaca']['api_secret']
            )
            
            logger.info("✅ Market data clients initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize market data clients: {e}")
            raise
    
    def add_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for data updates"""
        self.callbacks.append(callback)
    
    async def _notify_callbacks(self, data_type: str, data: Dict[str, Any]):
        """Notify all callbacks of new data"""
        for callback in self.callbacks:
            try:
                await callback(data_type, data)
            except Exception as e:
                logger.error(f"❌ Callback error: {e}")
    
    async def _handle_bar(self, bar):
        """Handle incoming bar data"""
        try:
            await self.buffer.add_bar(bar)
            await self._notify_callbacks("bar", {
                "symbol": bar.symbol,
                "timestamp": bar.timestamp.isoformat(),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume)
            })
            
        except Exception as e:
            logger.error(f"❌ Error handling bar data: {e}")
    
    async def _handle_quote(self, quote):
        """Handle incoming quote data"""
        try:
            await self.buffer.add_quote(quote)
            await self._notify_callbacks("quote", {
                "symbol": quote.symbol,
                "timestamp": quote.timestamp.isoformat(),
                "bid_price": float(quote.bid_price),
                "ask_price": float(quote.ask_price),
                "bid_size": float(quote.bid_size),
                "ask_size": float(quote.ask_size)
            })
            
        except Exception as e:
            logger.error(f"❌ Error handling quote data: {e}")
    
    async def _handle_trade(self, trade):
        """Handle incoming trade data"""
        try:
            await self.buffer.add_trade(trade)
            await self._notify_callbacks("trade", {
                "symbol": trade.symbol,
                "timestamp": trade.timestamp.isoformat(),
                "price": float(trade.price),
                "size": float(trade.size),
                "trade_id": trade.trade_id
            })
            
        except Exception as e:
            logger.error(f"❌ Error handling trade data: {e}")
    
    async def start_streaming(self, symbols: List[str]):
        """Start streaming data for given symbols"""
        try:
            if not self.stream_client:
                await self.initialize()
            
            # Subscribe to data streams
            self.stream_client.subscribe_bars(self._handle_bar, *symbols)
            self.stream_client.subscribe_quotes(self._handle_quote, *symbols)
            self.stream_client.subscribe_trades(self._handle_trade, *symbols)
            
            self.running = True
            logger.info(f"✅ Started streaming data for symbols: {symbols}")
            
            # Start the stream in a separate task
            import asyncio
            self._stream_task = asyncio.create_task(self._run_stream())
            
        except Exception as e:
            logger.error(f"❌ Error starting data stream: {e}")
            self.running = False
            raise
    
    async def _run_stream(self):
        """Run the data stream in a separate task"""
        try:
            # Start the stream
            await self.stream_client._run_forever()
        except Exception as e:
            logger.error(f"❌ Stream error: {e}")
            self.running = False
    
    async def stop_streaming(self):
        """Stop data streaming"""
        try:
            if self.stream_client and self.running:
                self.stream_client.stop()
                self.running = False
                
                # Cancel the stream task if it exists
                if hasattr(self, '_stream_task') and self._stream_task:
                    self._stream_task.cancel()
                    try:
                        await self._stream_task
                    except asyncio.CancelledError:
                        pass
                
                logger.info("✅ Data streaming stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping data stream: {e}")
    
    async def get_historical_data(self, symbol: str, timeframe: str = "1Min", limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical data for a symbol"""
        try:
            if not self.data_client:
                await self.initialize()
            
            # Map timeframe string to TimeFrame object
            timeframe_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, "Minute"),
                "15Min": TimeFrame(15, "Minute"),
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day
            }
            
            tf = timeframe_map.get(timeframe, TimeFrame.Minute)
            
            # Get data from last 24 hours
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            
            request = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=tf,
                start=start_time,
                end=end_time,
                limit=limit
            )
            
            bars = self.data_client.get_crypto_bars(request)
            bars_df = bars.df
            
            # Convert to list of dictionaries
            historical_data = []
            for idx, row in bars_df.iterrows():
                historical_data.append({
                    "timestamp": idx[1].isoformat(),
                    "symbol": symbol,
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume']),
                    "trade_count": int(row['trade_count']),
                    "vwap": float(row['vwap'])
                })
            
            return historical_data
            
        except Exception as e:
            logger.error(f"❌ Error getting historical data: {e}")
            return []
    
    async def get_market_context(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market context for a symbol"""
        try:
            # Get recent bars
            bars = await self.buffer.get_recent_bars(symbol, 20)
            
            # Get recent quotes
            quotes = await self.buffer.get_recent_quotes(symbol, 10)
            
            # Get recent trades
            trades = await self.buffer.get_recent_trades(symbol, 10)
            
            # Get latest price
            latest_price = await self.buffer.get_latest_price(symbol)
            
            # Calculate basic technical indicators
            technical_indicators = self._calculate_technical_indicators(bars)
            
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "latest_price": latest_price,
                "bars": bars,
                "quotes": quotes,
                "trades": trades,
                "technical_indicators": technical_indicators
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting market context: {e}")
            return {"symbol": symbol, "error": str(e)}
    
    def _calculate_technical_indicators(self, bars: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate basic technical indicators"""
        if len(bars) < 2:
            return {}
        
        try:
            closes = [bar["close"] for bar in bars]
            volumes = [bar["volume"] for bar in bars]
            
            # Simple Moving Average (SMA)
            sma_5 = sum(closes[-5:]) / min(5, len(closes)) if len(closes) >= 5 else closes[-1]
            sma_20 = sum(closes[-20:]) / min(20, len(closes)) if len(closes) >= 20 else closes[-1]
            
            # Price change
            price_change = closes[-1] - closes[-2] if len(closes) >= 2 else 0
            price_change_pct = (price_change / closes[-2]) * 100 if len(closes) >= 2 and closes[-2] != 0 else 0
            
            # Volume analysis
            avg_volume = sum(volumes) / len(volumes) if volumes else 0
            current_volume = volumes[-1] if volumes else 0
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Volatility (simple calculation)
            if len(closes) >= 10:
                returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, min(10, len(closes)))]
                volatility = (sum([r**2 for r in returns]) / len(returns))**0.5 * 100
            else:
                volatility = 0
            
            return {
                "sma_5": round(sma_5, 2),
                "sma_20": round(sma_20, 2),
                "price_change": round(price_change, 2),
                "price_change_pct": round(price_change_pct, 2),
                "volume_ratio": round(volume_ratio, 2),
                "volatility": round(volatility, 2),
                "trend": "bullish" if sma_5 > sma_20 else "bearish" if sma_5 < sma_20 else "neutral"
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating technical indicators: {e}")
            return {}


class MarketDataProcessor:
    """Process and analyze market data"""
    
    def __init__(self, buffer: MarketDataBuffer):
        self.buffer = buffer
        self.indicators = {}
    
    async def process_data_update(self, data_type: str, data: Dict[str, Any]):
        """Process incoming data updates"""
        try:
            symbol = data.get("symbol")
            if not symbol:
                return
            
            # Update indicators
            await self._update_indicators(symbol, data_type, data)
            
            # Check for trading signals
            await self._check_trading_signals(symbol)
            
        except Exception as e:
            logger.error(f"❌ Error processing data update: {e}")
    
    async def _update_indicators(self, symbol: str, data_type: str, data: Dict[str, Any]):
        """Update technical indicators"""
        try:
            if symbol not in self.indicators:
                self.indicators[symbol] = {
                    "price_history": [],
                    "volume_history": [],
                    "last_update": None
                }
            
            if data_type == "bar":
                self.indicators[symbol]["price_history"].append(data["close"])
                self.indicators[symbol]["volume_history"].append(data["volume"])
                self.indicators[symbol]["last_update"] = data["timestamp"]
                
                # Keep only last 100 values
                if len(self.indicators[symbol]["price_history"]) > 100:
                    self.indicators[symbol]["price_history"] = self.indicators[symbol]["price_history"][-100:]
                    self.indicators[symbol]["volume_history"] = self.indicators[symbol]["volume_history"][-100:]
            
        except Exception as e:
            logger.error(f"❌ Error updating indicators: {e}")
    
    async def _check_trading_signals(self, symbol: str):
        """Check for trading signals based on indicators"""
        try:
            if symbol not in self.indicators:
                return
            
            indicators = self.indicators[symbol]
            price_history = indicators["price_history"]
            
            if len(price_history) < 10:
                return
            
            # Simple signal detection (can be enhanced)
            recent_prices = price_history[-5:]
            older_prices = price_history[-10:-5]
            
            recent_avg = sum(recent_prices) / len(recent_prices)
            older_avg = sum(older_prices) / len(older_prices)
            
            if recent_avg > older_avg * 1.02:  # 2% increase
                logger.info(f"📈 Potential bullish signal for {symbol}")
            elif recent_avg < older_avg * 0.98:  # 2% decrease
                logger.info(f"📉 Potential bearish signal for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error checking trading signals: {e}")


async def main():
    """Test the market data pipeline"""
    # Initialize components
    streamer = MarketDataStreamer()
    processor = MarketDataProcessor(streamer.buffer)
    
    # Add processor as callback
    streamer.add_callback(processor.process_data_update)
    
    # Initialize and start streaming
    await streamer.initialize()
    
    # Test symbols
    symbols = ["BTC/USD", "ETH/USD"]
    
    try:
        # Start streaming
        await streamer.start_streaming(symbols)
    except KeyboardInterrupt:
        logger.info("Stopping data stream...")
        await streamer.stop_streaming()


if __name__ == "__main__":
    asyncio.run(main())
