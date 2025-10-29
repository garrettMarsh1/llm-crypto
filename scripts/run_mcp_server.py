"""
Script to run the Alpaca MCP server
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from mcp_servers.alpaca_server import AlpacaMCPServer
from loguru import logger


async def main():
    """Main function to run MCP server"""
    parser = argparse.ArgumentParser(description="Run Alpaca MCP Server")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    
    args = parser.parse_args()
    
    try:
        logger.info("🚀 Starting Alpaca MCP Server...")
        
        # Initialize server
        server = AlpacaMCPServer(args.config)
        await server.initialize()
        
        logger.info(f"✅ Alpaca MCP Server running on port {args.port}")
        logger.info("Press Ctrl+C to stop the server")
        
        # Run the server
        await server.server.run()
        
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
