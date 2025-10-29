@echo off
echo ========================================
echo Crypto Trading Agent - Windows Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not available
    pause
    exit /b 1
)

echo ✅ pip found

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv crypto_trading_env
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment created

REM Activate virtual environment
echo.
echo Activating virtual environment...
call crypto_trading_env\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment activated

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ERROR: Failed to upgrade pip
    pause
    exit /b 1
)

echo ✅ pip upgraded

REM Install PyTorch with CUDA support
echo.
echo Installing PyTorch with CUDA 12.1 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch with CUDA
    echo Trying CUDA 11.8 fallback...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install PyTorch with CUDA fallback
        pause
        exit /b 1
    )
)

echo ✅ PyTorch with CUDA installed

REM Install other ML dependencies
echo.
echo Installing ML dependencies...
pip install transformers>=4.40.0
pip install peft>=0.8.0
pip install bitsandbytes>=0.42.0
pip install accelerate>=0.25.0
pip install datasets>=2.16.0
if %errorlevel% neq 0 (
    echo ERROR: Failed to install ML dependencies
    pause
    exit /b 1
)

echo ✅ ML dependencies installed

REM Install trading and data dependencies
echo.
echo Installing trading and data dependencies...
pip install alpaca-py>=0.20.0
pip install websockets>=12.0
pip install loguru>=0.7.0
pip install backtrader>=1.9.78
if %errorlevel% neq 0 (
    echo ERROR: Failed to install trading dependencies
    pause
    exit /b 1
)

echo ✅ Trading dependencies installed

REM Install data science dependencies
echo.
echo Installing data science dependencies...
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0
pip install matplotlib>=3.7.0
pip install plotly>=5.15.0
if %errorlevel% neq 0 (
    echo ERROR: Failed to install data science dependencies
    pause
    exit /b 1
)

echo ✅ Data science dependencies installed

REM Install utility dependencies
echo.
echo Installing utility dependencies...
pip install python-dotenv>=1.0.0
pip install pydantic>=2.0.0
pip install fastapi>=0.100.0
pip install uvicorn>=0.23.0
pip install mcp>=0.1.0
if %errorlevel% neq 0 (
    echo ERROR: Failed to install utility dependencies
    pause
    exit /b 1
)

echo ✅ Utility dependencies installed

REM Verify CUDA installation
echo.
echo Verifying CUDA installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU count:', torch.cuda.device_count()); print('Current GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
if %errorlevel% neq 0 (
    echo ERROR: Failed to verify CUDA installation
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To activate the environment in the future, run:
echo   crypto_trading_env\Scripts\activate.bat
echo.
echo To test the installation, run:
echo   python quick_start.py
echo.
echo To start trading, run:
echo   python main.py
echo.
pause
