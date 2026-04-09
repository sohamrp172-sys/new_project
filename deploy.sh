#!/bin/bash
# Deployment script for full pipeline

set -e

echo "========================================"
echo "TELCO CHURN ML OPS - FULL DEPLOYMENT"
echo "========================================"

# Step 1: Install dependencies
echo "[1/6] Installing dependencies..."
python -m pip install -q -r requirements.txt

# Step 2: Run training pipeline
echo "[2/6] Training models..."
python -m src.models.train

# Step 3: Run tests
echo "[3/6] Running tests..."
python -m pytest tests/ -q --tb=short

# Step 4: Start monitoring in background
echo "[4/6] Starting monitoring..."
python monitor.py &
MONITOR_PID=$!

# Step 5: Start API server
echo "[5/6] Starting API server on http://localhost:8000..."
echo "      - API Docs: http://localhost:8000/docs"
echo "      - Swagger: http://localhost:8000/redoc"
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Step 6: Deployment success
echo "[6/6] Deployment complete!"
echo ""
echo "========================================"
echo "DEPLOYMENT STATUS: ✓ SUCCESS"
echo "========================================"
echo ""
echo "Services running:"
echo "  - API Server: PID $API_PID (port 8000)"
echo "  - Monitor: PID $MONITOR_PID"
echo ""
echo "To stop services:"
echo "  kill $API_PID $MONITOR_PID"
echo ""

# Keep script running
wait $API_PID $MONITOR_PID
