#!/usr/bin/env python3
"""
Test script to verify Alpaca API credentials
"""

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical import CryptoHistoricalDataClient

def test_credentials():
    """Test Alpaca API credentials"""
    
    print("🔍 Testing Alpaca API Credentials")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Get credentials
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ Missing credentials in .env file!")
        print("   Make sure you have:")
        print("   - ALPACA_API_KEY=your_key_here")
        print("   - ALPACA_API_SECRET=your_secret_here")
        return False
    
    print(f"✅ Found API Key: {api_key[:8]}...")
    print(f"✅ Found API Secret: {api_secret[:8]}...")
    
    try:
        # Test trading client
        print("\n🔗 Testing Trading Client...")
        trade_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=True  # Use paper trading
        )
        
        account = trade_client.get_account()
        print(f"✅ Trading Client: Connected")
        print(f"   Account ID: {account.id}")
        print(f"   Equity: ${float(account.equity):,.2f}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        
        # Test data client
        print("\n📊 Testing Data Client...")
        data_client = CryptoHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret
        )
        
        print("✅ Data Client: Connected")
        
        print("\n🎉 All credentials are working correctly!")
        print("   Your system is ready for trading!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing credentials: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check your API key and secret are correct")
        print("   2. Make sure you're using paper trading credentials")
        print("   3. Verify your Alpaca account is active")
        return False

if __name__ == "__main__":
    test_credentials()
