#!/usr/bin/env bash
# ============================================================
# CartComplete — One-shot setup script (Linux / macOS)
# ============================================================
set -e

echo "=== CartComplete Setup ==="

# 1. Create virtual environment
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/4] Virtual environment already exists."
fi

# 2. Activate
echo "[2/4] Activating virtual environment..."
source venv/bin/activate

# 3. Upgrade pip
echo "[3/4] Upgrading pip..."
pip install --upgrade pip

# 4. Install requirements
echo "[4/4] Installing requirements (this may take a few minutes)..."
pip install -r requirements.txt

echo ""
echo "=== Setup complete! ==="
echo "Next steps:"
echo "  1. Activate env  :  source venv/bin/activate"
echo "  2. Run tests     :  python -m pytest tests/ -v"
echo "  3. Start server  :  uvicorn src.serving.app:app --reload"
echo ""
