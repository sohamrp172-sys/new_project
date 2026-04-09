"""
Monitoring module for detecting data drift and model performance degradation.
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


class DataDriftMonitor:
    """
    Monitors for data drift by comparing feature distributions.
    """
    
    def __init__(self, baseline_stats: dict = None, drift_threshold: float = 0.1):
        """
        Initialize monitor.
        
        Args:
            baseline_stats: Dictionary of baseline statistics
            drift_threshold: Threshold for drift detection (as fraction of baseline)
        """
        self.baseline_stats = baseline_stats or {}
        self.drift_threshold = drift_threshold
        self.drift_report = {}
    
    def compute_baseline_stats(self, df: pd.DataFrame) -> dict:
        """
        Compute statistics from baseline data.
        
        Args:
            df: Training data
            
        Returns:
            Dictionary of statistics
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target if present
        if 'Churn' in numeric_cols:
            numeric_cols.remove('Churn')
        
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median())
            }
        
        self.baseline_stats = stats
        
        print(f"Computed baseline statistics for {len(stats)} features")
        return stats
    
    def detect_drift(self, df: pd.DataFrame, feature_name: str = None) -> dict:
        """
        Detect data drift in new data.
        
        Args:
            df: New data to check
            feature_name: Specific feature to check (optional)
            
        Returns:
            Dictionary of drift detection results
        """
        if not self.baseline_stats:
            raise ValueError("No baseline statistics available")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Churn' in numeric_cols:
            numeric_cols.remove('Churn')
        
        if feature_name and feature_name not in numeric_cols:
            raise ValueError(f"Feature {feature_name} not found or not numeric")
        
        features_to_check = [feature_name] if feature_name else numeric_cols
        
        drift_detected = False
        drift_details = {}
        
        for col in features_to_check:
            if col not in self.baseline_stats:
                continue
            
            baseline = self.baseline_stats[col]
            current_mean = float(df[col].mean())
            baseline_mean = baseline['mean']
            
            # Calculate drift as percentage change
            if baseline_mean != 0:
                drift_pct = abs(current_mean - baseline_mean) / abs(baseline_mean)
            else:
                drift_pct = 0
            
            is_drift = drift_pct > self.drift_threshold
            
            drift_details[col] = {
                'baseline_mean': baseline_mean,
                'current_mean': current_mean,
                'drift_percentage': float(drift_pct),
                'is_drift': is_drift,
                'threshold': self.drift_threshold
            }
            
            if is_drift:
                drift_detected = True
                print(f"\n⚠️  DRIFT DETECTED in {col}!")
                print(f"   Baseline mean: {baseline_mean:.4f}")
                print(f"   Current mean:  {current_mean:.4f}")
                print(f"   Drift: {drift_pct*100:.2f}%")
        
        self.drift_report = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': drift_detected,
            'features_analyzed': len(drift_details),
            'features_with_drift': sum(1 for v in drift_details.values() if v['is_drift']),
            'details': drift_details
        }
        
        return self.drift_report
    
    def save_report(self, save_path: str):
        """
        Save drift report to file.
        
        Args:
            save_path: Path to save report
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.drift_report, f, indent=4)
        
        print(f"\nDrift report saved to {save_path}")


def monitor_production_data(baseline_df: pd.DataFrame, new_df: pd.DataFrame,
                           experiments_path: str = "experiments",
                           drift_threshold: float = 0.1) -> dict:
    """
    Monitor new production data for drift.
    
    Args:
        baseline_df: Training data (baseline)
        new_df: New production data
        experiments_path: Path to save monitoring results
        drift_threshold: Drift threshold
        
    Returns:
        Dictionary of monitoring results
    """
    monitor = DataDriftMonitor(drift_threshold=drift_threshold)
    
    # Compute baseline from training data
    monitor.compute_baseline_stats(baseline_df)
    
    # Check new data
    print("\n" + "="*50)
    print("Monitoring New Data for Drift...")
    print("="*50)
    
    report = monitor.detect_drift(new_df)
    
    # Save report
    report_path = os.path.join(experiments_path, 'drift_report.json')
    monitor.save_report(report_path)
    
    return report


if __name__ == "__main__":
    print("Data drift monitoring module")
