# Activate Crypto Trading Environment
Write-Host "Activating Crypto Trading Environment..." -ForegroundColor Green
& ".\crypto_trading_env\Scripts\Activate.ps1"
Write-Host "Environment activated! You can now run:" -ForegroundColor Cyan
Write-Host "  python quick_start.py" -ForegroundColor White
Write-Host "  python main.py" -ForegroundColor White
Write-Host ""
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
Write-Host "Virtual environment: crypto_trading_env" -ForegroundColor Yellow
