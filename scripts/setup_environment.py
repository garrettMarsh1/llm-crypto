"""
Environment setup script for the crypto trading agent
"""

import os
import sys
import subprocess
from pathlib import Path
import yaml


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def check_gpu():
    """Check if CUDA GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ CUDA GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            print("⚠️ No CUDA GPU detected - will use CPU (slower)")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed - cannot check GPU")
        return False


def create_directories():
    """Create necessary directories"""
    directories = [
        "logs",
        "models",
        "data",
        "mcp_servers",
        "agents",
        "risk",
        "monitoring",
        "scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def create_env_file():
    """Create .env file from template"""
    env_file = Path(".env")
    env_example = Path("env_example.txt")
    
    if not env_file.exists() and env_example.exists():
        env_file.write_text(env_example.read_text())
        print("✅ Created .env file from template")
        print("⚠️ Please edit .env with your API credentials")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️ No env_example.txt found - please create .env manually")


def check_config():
    """Check if config.yaml exists and is valid"""
    config_file = Path("config.yaml")
    
    if not config_file.exists():
        print("❌ config.yaml not found")
        return False
    
    try:
        with open(config_file, 'r') as f:
            yaml.safe_load(f)
        print("✅ config.yaml is valid")
        return True
    except yaml.YAMLError as e:
        print(f"❌ config.yaml is invalid: {e}")
        return False


def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_api_credentials():
    """Check if API credentials are set"""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("⚠️ .env file not found - please create it with your API credentials")
        return False
    
    required_vars = ["ALPACA_API_KEY", "ALPACA_API_SECRET"]
    missing_vars = []
    
    with open(env_file, 'r') as f:
        content = f.read()
        for var in required_vars:
            if f"{var}=" not in content or f"{var}=your_" in content:
                missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️ Missing or placeholder API credentials: {', '.join(missing_vars)}")
        print("Please edit .env with your actual API credentials")
        return False
    
    print("✅ API credentials configured")
    return True


def test_imports():
    """Test if all required modules can be imported"""
    required_modules = [
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "alpaca",
        "loguru",
        "pandas",
        "numpy"
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing dependencies")
        return False
    
    print("✅ All required modules imported successfully")
    return True


def main():
    """Main setup function"""
    print("🚀 Setting up Crypto Trading Agent Environment")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Check config
    config_valid = check_config()
    
    # Install dependencies
    deps_installed = install_dependencies()
    
    # Check API credentials
    api_configured = check_api_credentials()
    
    # Test imports
    imports_ok = test_imports()
    
    print("\n" + "=" * 50)
    print("SETUP SUMMARY")
    print("=" * 50)
    
    if gpu_available:
        print("✅ GPU: CUDA available")
    else:
        print("⚠️ GPU: CUDA not available (will use CPU)")
    
    print(f"{'✅' if config_valid else '❌'} Config: config.yaml")
    print(f"{'✅' if deps_installed else '❌'} Dependencies: Installed")
    print(f"{'✅' if api_configured else '⚠️'} API: Credentials configured")
    print(f"{'✅' if imports_ok else '❌'} Imports: All modules available")
    
    if all([config_valid, deps_installed, imports_ok]):
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env with your Alpaca API credentials")
        print("2. Run: python main.py --mode paper --symbols BTC/USD")
        print("3. Monitor with: python scripts/monitor_dashboard.py")
    else:
        print("\n❌ Setup incomplete - please fix the issues above")
        sys.exit(1)


if __name__ == "__main__":
    main()
