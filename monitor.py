"""
Monitoring script to check for data drift in production.
"""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.monitoring.monitor import monitor_production_data


def run_monitoring(baseline_path: str, new_data_path: str):
    """
    Run monitoring on production data.
    
    Args:
        baseline_path: Path to training data (baseline)
        new_data_path: Path to new production data
    """
    print("\n" + "="*70)
    print("DATA DRIFT MONITORING")
    print("="*70)
    
    try:
        # Load data
        baseline_df = pd.read_csv(baseline_path)
        new_df = pd.read_csv(new_data_path)
        
        print(f"Baseline data shape: {baseline_df.shape}")
        print(f"New data shape: {new_df.shape}")
        
        # Run monitoring
        report = monitor_production_data(
            baseline_df, new_df,
            experiments_path="experiments",
            drift_threshold=0.1
        )
        
        # Print summary
        print("\n" + "="*70)
        print("MONITORING SUMMARY")
        print("="*70)
        print(f"Drift Detected: {report['drift_detected']}")
        print(f"Features Analyzed: {report['features_analyzed']}")
        print(f"Features with Drift: {report['features_with_drift']}")
        
        if report['drift_detected']:
            print("\n⚠️  WARNING: Data drift detected!")
            print("Consider retraining the model with recent data.")
        else:
            print("\n✓ No significant data drift detected.")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Monitoring failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # In production, new_data_path should point to fresh incoming data.
    # Using the holdout set as a stand-in for demonstration.
    baseline_path = "data/processed/engineered_features.csv"
    new_data_path = "data/raw/cell2cellholdout.csv"  # different dataset as proxy for new data
    
    run_monitoring(baseline_path, new_data_path)
