@echo off
REM Deployment script for Windows PowerShell
REM Full pipeline: train, evaluate, register, test, deploy API

setlocal enabledelayedexpansion

echo.
echo ========================================
echo TELCO CHURN ML OPS - FULL DEPLOYMENT
echo ========================================
echo.

set PYTHON=C:/Users/Soham/AppData/Local/Microsoft/WindowsApps/python3.11.exe

REM Step 1: Install dependencies
echo [1/6] Installing dependencies...
%PYTHON% -m pip install -q -r requirements.txt
if errorlevel 1 goto :error

REM Step 2: Run training pipeline
echo [2/6] Training models and registering best...
%PYTHON% -m src.models.train
if errorlevel 1 goto :error

REM Step 3: Run tests
echo [3/6] Running test suite...
%PYTHON% -m pytest tests/ -q --tb=short
if errorlevel 1 goto :error

REM Step 4: Start monitoring
echo [4/6] Starting data drift monitoring...
start /B "Monitoring" %PYTHON% monitor.py

REM Step 5: Start API server
echo [5/6] Starting API server on http://localhost:8000...
echo       - API Docs: http://localhost:8000/docs
echo       - Swagger: http://localhost:8000/redoc
start "API Server" %PYTHON% -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

REM Step 6: Deployment success
echo.
echo ========================================
echo DEPLOYMENT STATUS: SUCCESS
echo ========================================
echo.
echo Services running:
echo   - API Server: http://localhost:8000
echo   - Monitoring: Background process
echo.
echo Access the API:
echo   - Health check: curl http://localhost:8000/health
echo   - Swagger UI: http://localhost:8000/docs
echo   - ReDoc: http://localhost:8000/redoc
echo   - Predict: curl -X POST http://localhost:8000/predict ^
echo              -H "Content-Type: application/json" ^
echo              -d "{\"tenure\": 24, \"monthlycharges\": 65.5, \"totalcharges\": 1574.5}"
echo.

goto :success

:error
echo.
echo ERROR: Deployment failed!
exit /b 1

:success
exit /b 0
