#!/usr/bin/env python
"""
Comprehensive deployment script for telco churn MLOps project.
Handles: training, evaluation, registration, testing, and API serving.
"""

import subprocess
import sys
import time
import os
from pathlib import Path


class Deployment:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.success = True
        self.processes = []

    def run_command(self, cmd, description, background=False):
        """Execute a shell command and report status."""
        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"{'='*60}")

        try:
            if background:
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    cwd=self.project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.processes.append(process)
                print(f"✓ Started in background (PID: {process.pid})")
                return True
            else:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=self.project_root,
                    capture_output=False
                )
                if result.returncode == 0:
                    print(f"✓ {description} completed successfully")
                    return True
                else:
                    print(f"✗ {description} failed with code {result.returncode}")
                    self.success = False
                    return False
        except Exception as e:
            print(f"✗ Error: {e}")
            self.success = False
            return False

    def deploy(self):
        """Execute full deployment pipeline."""
        print("\n")
        print("╔" + "="*58 + "╗")
        print("║" + " "*58 + "║")
        print("║" + "  TELCO CHURN ML OPS - FULL DEPLOYMENT".center(58) + "║")
        print("║" + " "*58 + "║")
        print("╚" + "="*58 + "╝")

        # Step 1: Install dependencies
        self.run_command(
            f"{sys.executable} -m pip install -q -r requirements.txt",
            "[1/6] Installing dependencies"
        )

        # Step 2: Train models
        self.run_command(
            f"{sys.executable} -m src.models.train",
            "[2/6] Training models and evaluating"
        )

        # Step 3: Run tests
        self.run_command(
            f"{sys.executable} -m pytest tests/ -q --tb=short",
            "[3/6] Running test suite"
        )

        # Step 4: Start monitoring (background)
        self.run_command(
            f"{sys.executable} monitor.py",
            "[4/6] Starting data drift monitoring",
            background=True
        )
        time.sleep(1)

        # Step 5: Start API server (background)
        self.run_command(
            f"{sys.executable} -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000",
            "[5/6] Starting API server",
            background=True
        )
        time.sleep(2)

        # Step 6: Summary
        print(f"\n{'='*60}")
        print("[6/6] Deployment Summary")
        print(f"{'='*60}")

        if self.success:
            print("\n✓ DEPLOYMENT SUCCESSFUL")
            print("\nServices running:")
            print("  • API Server: http://localhost:8000")
            print("  • Health Check: http://localhost:8000/health")
            print("  • API Docs: http://localhost:8000/docs")
            print("  • Swagger: http://localhost:8000/redoc")
            print("  • Data Monitoring: Active")
            print("\nExample API calls:")
            print('  curl http://localhost:8000/health')
            print('  curl -X POST http://localhost:8000/predict \\')
            print('    -H "Content-Type: application/json" \\')
            print('    -d \'{"tenure": 24, "monthlycharges": 65.5, "totalcharges": 1574.5}\'')
            print(f"\nTo stop services, press Ctrl+C")
            print(f"{'='*60}\n")

            # Keep script running
            try:
                for process in self.processes:
                    process.wait()
            except KeyboardInterrupt:
                print("\n\nShutting down services...")
                for process in self.processes:
                    process.terminate()
                print("✓ All services stopped")

            return 0
        else:
            print("\n✗ DEPLOYMENT FAILED")
            print("Check errors above and retry.")
            return 1


if __name__ == "__main__":
    deployer = Deployment()
    sys.exit(deployer.deploy())
