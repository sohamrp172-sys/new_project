"""
Setup script to help users get started quickly.
Copies sample data files if available.
"""
import os
import shutil
import sys
from pathlib import Path


def setup_project():
    """Initialize project structure."""
    
    print("\n" + "="*70)
    print("TELCO CHURN MLOPS PROJECT - SETUP WIZARD")
    print("="*70)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("\n⚠️  Warning: Python 3.10+ recommended")
        print(f"   Current: Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check directories
    print("\n✓ Checking project structure...")
    required_dirs = [
        'data/raw',
        'data/processed',
        'models',
        'experiments',
        'src/data',
        'src/features',
        'src/models',
        'src/api',
        'src/monitoring',
        'tests'
    ]
    
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✓ {dir_path}/")
    
    # Check for CSV files
    print("\n✓ Checking for data files...")
    raw_data_dir = Path('data/raw')
    csv_files = list(raw_data_dir.glob('*.csv'))
    
    if csv_files:
        print(f"  ✓ Found {len(csv_files)} CSV files:")
        for f in csv_files:
            print(f"    - {f.name}")
    else:
        print("  ⚠️  No CSV files found in data/raw/")
        print("     Please copy your churn datasets to data/raw/")
    
    # Check requirements
    print("\n✓ Checking requirements.txt...")
    if os.path.exists('requirements.txt'):
        print("  ✓ requirements.txt found")
        print("\n  To install dependencies, run:")
        print("    pip install -r requirements.txt")
    
    # Print next steps
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    
    print("\n1. SETUP ENVIRONMENT")
    print("   python -m venv venv")
    print("   # Windows:")
    print("   venv\\Scripts\\activate")
    print("   # Linux/Mac:")
    print("   source venv/bin/activate")
    
    print("\n2. INSTALL DEPENDENCIES")
    print("   pip install -r requirements.txt")
    
    if not csv_files:
        print("\n3. ADD DATA FILES")
        print("   Copy your CSV files to: data/raw/")
        print("   Required: 7 telco churn datasets")
    else:
        print("\n3. TRAIN MODEL")
        print("   python train.py")
        
        print("\n4. START API")
        print("   python -m uvicorn src.api.main:app --reload")
        
        print("\n5. TEST PREDICTIONS")
        print("   curl -X POST http://localhost:8000/predict \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d '{\"tenure\": 24, \"monthlycharges\": 65.50, \"totalcharges\": 1574.50}'")
    
    print("\n6. MONITOR DATA DRIFT")
    print("   python monitor.py")
    
    print("\n" + "="*70)
    print("DOCUMENTATION")
    print("="*70)
    print("\nRead README.md for detailed instructions and documentation")
    print("Read summary.txt for project overview and architecture")
    
    print("\n" + "="*70 + "\n")


def check_imports():
    """Check if all required packages are available."""
    print("Checking Python dependencies...")
    
    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'fastapi',
        'uvicorn',
        'pydantic',
        'joblib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


if __name__ == "__main__":
    setup_project()
    
    # Optionally check imports
    print("\nWould you like to check Python dependencies? (y/n)")
    response = input().strip().lower()
    if response == 'y':
        check_imports()
