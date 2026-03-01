# ============================================================
# CartComplete — One-shot setup script (Windows PowerShell)
# ============================================================

Write-Host "=== CartComplete Setup ===" -ForegroundColor Cyan

# 1. Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "[1/4] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
} else {
    Write-Host "[1/4] Virtual environment already exists." -ForegroundColor Green
}

# 2. Activate
Write-Host "[2/4] Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# 3. Upgrade pip
Write-Host "[3/4] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 4. Install requirements
Write-Host "[4/4] Installing requirements (this may take a few minutes)..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "=== Setup complete! ===" -ForegroundColor Green
Write-Host "Next steps:"
Write-Host "  1. Activate env  :  .\venv\Scripts\Activate.ps1"
Write-Host "  2. Run tests     :  python -m pytest tests/ -v"
Write-Host "  3. Start server  :  uvicorn src.serving.app:app --reload"
Write-Host ""
