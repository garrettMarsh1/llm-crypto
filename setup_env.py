#!/usr/bin/env python3
"""
Setup script to create .env file for Alpaca API credentials
"""

def create_env_file():
    """Create .env file with Alpaca API credentials"""
    
    print("🔧 Setting up .env file for Alpaca API credentials")
    print("=" * 50)
    
    # Get API credentials from user
    api_key = input("Enter your Alpaca API Key: ").strip()
    api_secret = input("Enter your Alpaca API Secret: ").strip()
    
    if not api_key or not api_secret:
        print("❌ API Key and Secret are required!")
        return False
    
    # Create .env content
    env_content = f"""# Alpaca API Configuration
ALPACA_API_KEY={api_key}
ALPACA_API_SECRET={api_secret}

# Optional: Override other settings
# ALPACA_BASE_URL=https://paper-api.alpaca.markets
# ALPACA_DATA_URL=https://data.alpaca.markets
"""
    
    # Write .env file
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✅ .env file created successfully!")
        print("📁 Location: .env")
        print("\n🔒 Security Note: Never commit .env file to version control!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

if __name__ == "__main__":
    create_env_file()
