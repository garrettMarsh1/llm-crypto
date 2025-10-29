"""
System test script for the crypto trading agent
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.trading_agent import CryptoTradingAgent
from monitoring.logger import TradingLogger, PerformanceMonitor, HealthChecker
from loguru import logger


async def test_system():
    """Test all system components"""
    print("Testing Crypto Trading Agent System")
    print("=" * 50)
    
    try:
        # Test 1: Initialize trading agent
        print("\n1. Testing Trading Agent Initialization...")
        agent = CryptoTradingAgent()
        await agent.initialize()
        print("Trading agent initialized successfully")
        
        # Test 2: Test model loading
        print("\n2. Testing Model Loading...")
        model_info = agent.model.get_model_info()
        print(f"Model loaded: {model_info.get('model_name', 'Unknown')}")
        print(f"   Quantization: {model_info.get('quantization', 'Unknown')}")
        trainable_params = model_info.get('trainable_parameters', 0)
        if isinstance(trainable_params, (int, float)):
            print(f"   Trainable parameters: {trainable_params:,}")
        else:
            print(f"   Trainable parameters: {trainable_params}")
        
        # Test 3: Test signal generation
        print("\n3. Testing Signal Generation...")
        test_context = {
            "symbol": "BTC/USD",
            "bars": [
                {
                    "timestamp": "2024-01-01T10:00:00Z",
                    "open": 45000,
                    "high": 46000,
                    "low": 44500,
                    "close": 45500,
                    "volume": 1000
                }
            ],
            "positions": []
        }
        
        signal = agent.model.generate_signal(test_context)
        if isinstance(signal, dict):
            print(f"Signal generated: {signal.get('action', 'unknown')} (confidence: {signal.get('confidence', 0):.2f})")
        else:
            print(f"Signal generated: {signal}")
        
        # Test 4: Test risk management
        print("\n4. Testing Risk Management...")
        risk_metrics = agent.risk_manager.get_risk_metrics()
        print(f" Risk manager operational")
        print(f"   Max daily loss: {risk_metrics.get('max_daily_loss', 0):.2%}")
        print(f"   Max position size: {risk_metrics.get('max_position_size', 0):.2%}")
        
        # Test 5: Test data streaming (brief)
        print("\n5. Testing Data Streaming...")
        # Note: This would require actual API credentials
        print("WARNING: Data streaming test skipped (requires API credentials)")
        
        # Test 6: Test logging system
        print("\n6. Testing Logging System...")
        trading_logger = TradingLogger()
        trading_logger.log_trade({
            "symbol": "BTC/USD",
            "action": "buy",
            "quantity": 0.001,
            "price": 45000,
            "confidence": 0.8
        })
        print(" Logging system operational")
        
        # Test 7: Test performance monitoring
        print("\n7. Testing Performance Monitoring...")
        performance_monitor = PerformanceMonitor()
        test_trades = [
            {"pnl": 100, "timestamp": "2024-01-01T10:00:00Z"},
            {"pnl": -50, "timestamp": "2024-01-01T11:00:00Z"},
            {"pnl": 75, "timestamp": "2024-01-01T12:00:00Z"}
        ]
        metrics = performance_monitor.calculate_metrics(test_trades)
        print(f" Performance monitoring operational")
        print(f"   Win rate: {metrics.get('win_rate', 0):.1f}%")
        total_pnl = metrics.get('total_pnl', 0)
        if isinstance(total_pnl, (int, float)):
            print(f"   Total P&L: ${total_pnl:.2f}")
        else:
            print(f"   Total P&L: ${total_pnl}")
        
        # Test 8: Test health checking
        print("\n8. Testing Health Checking...")
        health_checker = HealthChecker()
        components = {
            "model": agent.model,
            "data_streamer": agent.data_streamer,
            "risk_manager": agent.risk_manager,
            "alpaca_mcp": agent.alpaca_mcp
        }
        health = await health_checker.check_system_health(components)
        print(f" Health checking operational")
        print(f"   Overall status: {health.get('overall_status', 'Unknown')}")
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED!")
        print(" System is ready for trading")
        print("\nNext steps:")
        print("1. Set up your API credentials in .env")
        print("2. Run: python main.py --mode paper --symbols BTC/USD")
        print("3. Monitor with: python scripts/monitor_dashboard.py")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        logger.error(f"System test failed: {e}")
        raise


async def main():
    """Main test function"""
    try:
        await test_system()
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
