"""
Alpaca Trading MCP Server
Provides trading operations through Model Context Protocol
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.live import CryptoDataStream
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

# MCP imports - these will be available when MCP is installed
try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
    from mcp.server.models import InitializationOptions
    import mcp.types as types
    MCP_AVAILABLE = True
except ImportError:
    # Fallback for when MCP is not installed
    print("⚠️ MCP not installed. Install with: pip install mcp")
    MCP_AVAILABLE = False
    # Create dummy classes for development
    class Server:
        def __init__(self, name): 
            self.name = name
            self._tools = []
            self._handlers = {}
        def list_tools(self): 
            def decorator(func):
                self._tools = func()
                return func
            return decorator
        def call_tool(self): 
            def decorator(func):
                self._handlers['call_tool'] = func
                return func
            return decorator
        def get_capabilities(self, **kwargs):
            return {"tools": {"listChanged": True}}
        async def run(self, read_stream, write_stream, init_options):
            print("MCP server running (mock mode)")
    class Tool:
        def __init__(self, **kwargs): 
            self.__dict__.update(kwargs)
    class TextContent:
        def __init__(self, **kwargs): 
            self.__dict__.update(kwargs)
    class InitializationOptions:
        def __init__(self, **kwargs): 
            self.__dict__.update(kwargs)
    import types


class AlpacaMCPServer:
    """MCP Server for Alpaca trading operations"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.trade_client = None
        self.data_client = None
        self.stream_client = None
        self.server = Server("alpaca-trading-server")
        self._setup_handlers()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file and substitute environment variables"""
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Substitute environment variables
        config_content = os.path.expandvars(config_content)
        
        return yaml.safe_load(config_content)
    
    def _setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available trading tools"""
            return [
                Tool(
                    name="get_account_info",
                    description="Get account information and balances",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="get_positions",
                    description="Get current positions",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="get_orders",
                    description="Get recent orders",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["open", "closed", "all"],
                                "description": "Filter orders by status"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of orders to return",
                                "default": 50
                            }
                        }
                    }
                ),
                Tool(
                    name="place_market_order",
                    description="Place a market order",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol (e.g., BTC/USD)"
                            },
                            "side": {
                                "type": "string",
                                "enum": ["buy", "sell"],
                                "description": "Order side"
                            },
                            "quantity": {
                                "type": "number",
                                "description": "Order quantity"
                            }
                        },
                        "required": ["symbol", "side", "quantity"]
                    }
                ),
                Tool(
                    name="place_limit_order",
                    description="Place a limit order",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol (e.g., BTC/USD)"
                            },
                            "side": {
                                "type": "string",
                                "enum": ["buy", "sell"],
                                "description": "Order side"
                            },
                            "quantity": {
                                "type": "number",
                                "description": "Order quantity"
                            },
                            "limit_price": {
                                "type": "number",
                                "description": "Limit price"
                            }
                        },
                        "required": ["symbol", "side", "quantity", "limit_price"]
                    }
                ),
                Tool(
                    name="cancel_order",
                    description="Cancel an order",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "order_id": {
                                "type": "string",
                                "description": "Order ID to cancel"
                            }
                        },
                        "required": ["order_id"]
                    }
                ),
                Tool(
                    name="get_crypto_bars",
                    description="Get historical crypto bars",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Trading symbol (e.g., BTC/USD)"
                            },
                            "timeframe": {
                                "type": "string",
                                "enum": ["1Min", "5Min", "15Min", "1Hour", "1Day"],
                                "description": "Bar timeframe"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of bars to return",
                                "default": 100
                            }
                        },
                        "required": ["symbol", "timeframe"]
                    }
                ),
                Tool(
                    name="get_crypto_quotes",
                    description="Get latest crypto quotes",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbols": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of symbols to get quotes for"
                            }
                        },
                        "required": ["symbols"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool calls"""
            try:
                if name == "get_account_info":
                    return await self._get_account_info()
                elif name == "get_positions":
                    return await self._get_positions()
                elif name == "get_orders":
                    return await self._get_orders(arguments)
                elif name == "place_market_order":
                    return await self._place_market_order(arguments)
                elif name == "place_limit_order":
                    return await self._place_limit_order(arguments)
                elif name == "cancel_order":
                    return await self._cancel_order(arguments)
                elif name == "get_crypto_bars":
                    return await self._get_crypto_bars(arguments)
                elif name == "get_crypto_quotes":
                    return await self._get_crypto_quotes(arguments)
                else:
                    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def initialize(self):
        """Initialize Alpaca clients"""
        try:
            self.trade_client = TradingClient(
                api_key=self.config['alpaca']['api_key'],
                secret_key=self.config['alpaca']['api_secret'],
                paper=self.config['alpaca']['paper_trading']
            )
            
            self.data_client = CryptoHistoricalDataClient(
                api_key=self.config['alpaca']['api_key'],
                secret_key=self.config['alpaca']['api_secret']
            )
            
            print("Alpaca clients initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Alpaca clients: {e}")
            raise
    
    async def _get_account_info(self) -> List[types.TextContent]:
        """Get account information"""
        try:
            account = self.trade_client.get_account()
            info = {
                "account_id": account.id,
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "trading_blocked": account.trading_blocked,
                "pattern_day_trader": account.pattern_day_trader
            }
            return [types.TextContent(type="text", text=json.dumps(info, indent=2))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error getting account info: {e}")]
    
    async def _get_positions(self) -> List[types.TextContent]:
        """Get current positions"""
        try:
            positions = self.trade_client.get_all_positions()
            position_data = []
            for pos in positions:
                position_data.append({
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "side": pos.side,
                    "market_value": float(pos.market_value),
                    "cost_basis": float(pos.cost_basis),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc)
                })
            return [types.TextContent(type="text", text=json.dumps(position_data, indent=2))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error getting positions: {e}")]
    
    async def _get_orders(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Get recent orders"""
        try:
            status = args.get('status', 'all')
            limit = args.get('limit', 50)
            
            if status == 'all':
                orders = self.trade_client.get_orders(status='all', limit=limit)
            else:
                orders = self.trade_client.get_orders(status=status, limit=limit)
            
            order_data = []
            for order in orders:
                order_data.append({
                    "id": order.id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "qty": float(order.qty),
                    "filled_qty": float(order.filled_qty),
                    "order_type": order.order_type,
                    "status": order.status,
                    "created_at": order.created_at.isoformat(),
                    "filled_at": order.filled_at.isoformat() if order.filled_at else None,
                    "limit_price": float(order.limit_price) if order.limit_price else None
                })
            
            return [types.TextContent(type="text", text=json.dumps(order_data, indent=2))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error getting orders: {e}")]
    
    async def _place_market_order(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Place a market order"""
        try:
            symbol = args['symbol']
            side = OrderSide.BUY if args['side'] == 'buy' else OrderSide.SELL
            quantity = args['quantity']
            
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.GTC
            )
            
            order = self.trade_client.submit_order(order_request)
            
            result = {
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "qty": float(order.qty),
                "order_type": order.order_type,
                "status": order.status,
                "created_at": order.created_at.isoformat()
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error placing market order: {e}")]
    
    async def _place_limit_order(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Place a limit order"""
        try:
            symbol = args['symbol']
            side = OrderSide.BUY if args['side'] == 'buy' else OrderSide.SELL
            quantity = args['quantity']
            limit_price = args['limit_price']
            
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=side,
                limit_price=limit_price,
                time_in_force=TimeInForce.GTC
            )
            
            order = self.trade_client.submit_order(order_request)
            
            result = {
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "qty": float(order.qty),
                "limit_price": float(order.limit_price),
                "order_type": order.order_type,
                "status": order.status,
                "created_at": order.created_at.isoformat()
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error placing limit order: {e}")]
    
    async def _cancel_order(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Cancel an order"""
        try:
            order_id = args['order_id']
            self.trade_client.cancel_order_by_id(order_id)
            return [types.TextContent(type="text", text=f"Order {order_id} cancelled successfully")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error cancelling order: {e}")]
    
    async def _get_crypto_bars(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Get historical crypto bars"""
        try:
            symbol = args['symbol']
            timeframe_str = args['timeframe']
            limit = args.get('limit', 100)
            
            # Map timeframe string to TimeFrame object
            timeframe_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, "Minute"),
                "15Min": TimeFrame(15, "Minute"),
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day
            }
            
            timeframe = timeframe_map[timeframe_str]
            
            # Get bars from last 24 hours
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            
            request = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=start_time,
                end=end_time,
                limit=limit
            )
            
            bars = self.data_client.get_crypto_bars(request)
            bars_df = bars.df
            
            # Convert to list of dictionaries
            bars_data = []
            for idx, row in bars_df.iterrows():
                bars_data.append({
                    "timestamp": idx[1].isoformat(),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume']),
                    "trade_count": int(row['trade_count']),
                    "vwap": float(row['vwap'])
                })
            
            return [types.TextContent(type="text", text=json.dumps(bars_data, indent=2))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error getting crypto bars: {e}")]
    
    async def _get_crypto_quotes(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Get latest crypto quotes"""
        try:
            symbols = args['symbols']
            # Note: Alpaca doesn't have a direct quotes endpoint for crypto
            # We'll use the latest bar data as a proxy
            quotes_data = []
            
            for symbol in symbols:
                try:
                    request = CryptoBarsRequest(
                        symbol_or_symbols=[symbol],
                        timeframe=TimeFrame.Minute,
                        limit=1
                    )
                    bars = self.data_client.get_crypto_bars(request)
                    if not bars.df.empty:
                        latest_bar = bars.df.iloc[-1]
                        quotes_data.append({
                            "symbol": symbol,
                            "price": float(latest_bar['close']),
                            "timestamp": bars.df.index[-1][1].isoformat(),
                            "volume": float(latest_bar['volume'])
                        })
                except Exception as e:
                    quotes_data.append({
                        "symbol": symbol,
                        "error": str(e)
                    })
            
            return [types.TextContent(type="text", text=json.dumps(quotes_data, indent=2))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error getting crypto quotes: {e}")]


async def main():
    """Main function to run the MCP server"""
    server = AlpacaMCPServer()
    await server.initialize()
    
    # Run the server
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="alpaca-trading-server",
                server_version="1.0.0",
                capabilities=server.server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
