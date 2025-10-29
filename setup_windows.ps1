2# Crypto Trading Agent - Windows PowerShell Setup Script
Write-Host "========================================" -ForegroundColor Green
Write-Host "Crypto Trading Agent - Windows Setup" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if pip is available
try {
    $pipVersion = pip --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "pip not found"
    }
    Write-Host "✅ pip found" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: pip is not available" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
try {
    python -m venv crypto_trading_env
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment"
    }
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Failed to create virtual environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
try {
    & ".\crypto_trading_env\Scripts\Activate.ps1"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to activate virtual environment"
    }
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "You may need to run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
try {
    python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip"
    }
    Write-Host "✅ pip upgraded" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Failed to upgrade pip" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install PyTorch with CUDA support
Write-Host ""
Write-Host "Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Yellow
try {
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️ CUDA 12.1 failed, trying CUDA 11.8 fallback..." -ForegroundColor Yellow
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install PyTorch with CUDA"
        }
    }
    Write-Host "✅ PyTorch with CUDA installed" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Failed to install PyTorch with CUDA" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install ML dependencies
Write-Host ""
Write-Host "Installing ML dependencies..." -ForegroundColor Yellow
$mlDeps = @(
    "transformers>=4.40.0",
    "peft>=0.8.0", 
    "bitsandbytes>=0.42.0",
    "accelerate>=0.25.0",
    "datasets>=2.16.0"
)

foreach ($dep in $mlDeps) {
    try {
        pip install $dep
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install $dep"
        }
    } catch {
        Write-Host "❌ ERROR: Failed to install $dep" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}
Write-Host "✅ ML dependencies installed" -ForegroundColor Green

# Install trading dependencies
Write-Host ""
Write-Host "Installing trading dependencies..." -ForegroundColor Yellow
$tradingDeps = @(
    "alpaca-py>=0.20.0",
    "websockets>=12.0",
    "loguru>=0.7.0",
    "backtrader>=1.9.78"
)

foreach ($dep in $tradingDeps) {
    try {
        pip install $dep
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install $dep"
        }
    } catch {
        Write-Host "❌ ERROR: Failed to install $dep" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}
Write-Host "✅ Trading dependencies installed" -ForegroundColor Green

# Install data science dependencies
Write-Host ""
Write-Host "Installing data science dependencies..." -ForegroundColor Yellow
$dataDeps = @(
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "plotly>=5.15.0"
)

foreach ($dep in $dataDeps) {
    try {
        pip install $dep
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install $dep"
        }
    } catch {
        Write-Host "❌ ERROR: Failed to install $dep" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}
Write-Host "✅ Data science dependencies installed" -ForegroundColor Green

# Install utility dependencies
Write-Host ""
Write-Host "Installing utility dependencies..." -ForegroundColor Yellow
$utilDeps = @(
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "mcp>=0.1.0"
)

foreach ($dep in $utilDeps) {
    try {
        pip install $dep
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install $dep"
        }
    } catch {
        Write-Host "❌ ERROR: Failed to install $dep" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}
Write-Host "✅ Utility dependencies installed" -ForegroundColor Green

# Verify CUDA installation
Write-Host ""
Write-Host "Verifying CUDA installation..." -ForegroundColor Yellow
try {
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU count:', torch.cuda.device_count()); print('Current GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to verify CUDA installation"
    }
    Write-Host "✅ CUDA verification complete" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Failed to verify CUDA installation" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment in the future, run:" -ForegroundColor Cyan
Write-Host "  .\crypto_trading_env\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To test the installation, run:" -ForegroundColor Cyan
Write-Host "  python quick_start.py" -ForegroundColor White
Write-Host ""
Write-Host "To start trading, run:" -ForegroundColor Cyan
Write-Host "  python main.py" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to exit"
