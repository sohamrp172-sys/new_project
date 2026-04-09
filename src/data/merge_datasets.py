"""
Data merging module for combining multiple churn datasets.
"""
import pandas as pd
import os
import json
from pathlib import Path
from datetime import datetime


def load_all_datasets(raw_data_path: str) -> list:
    """
    Load all CSV files from raw data directory.
    
    Args:
        raw_data_path: Path to raw data directory
        
    Returns:
        List of DataFrames
    """
    datasets = []
    csv_files = list(Path(raw_data_path).glob("*.csv"))
    
    print(f"Found {len(csv_files)} CSV files in {raw_data_path}")
    
    for csv_file in csv_files:
        print(f"Loading {csv_file.name}...")
        df = pd.read_csv(csv_file)
        print(f"  Shape: {df.shape}")
        datasets.append(df)
    
    return datasets


def standardize_churn_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the Churn column to binary (0, 1).
    Handle various formats: 'Yes'/'No', 1/0, True/False, etc.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized Churn column
    """
    df = df.copy()
    
    # Find churn column (case-insensitive)
    churn_cols = [col for col in df.columns if 'churn' in col.lower()]
    
    if not churn_cols:
        print("Warning: No churn column found. Skipping standardization.")
        return df
    
    churn_col = churn_cols[0]
    
    # Convert to string and standardize
    df[churn_col] = df[churn_col].astype(str).str.lower()
    
    # Map to binary
    df[churn_col] = df[churn_col].map({
        'yes': 1, 'no': 0,
        '1': 1, '0': 0,
        'true': 1, 'false': 0,
        '1.0': 1, '0.0': 0
    })
    
    # Rename to standard name
    if churn_col != 'Churn':
        df.rename(columns={churn_col: 'Churn'}, inplace=True)
    
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names for consistency.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with normalized column names
    """
    df = df.copy()
    
    # Remove whitespace and standardize naming
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    
    return df


def merge_datasets(datasets: list) -> pd.DataFrame:
    """
    Merge multiple datasets into one.
    
    Args:
        datasets: List of DataFrames to merge
        
    Returns:
        Merged DataFrame
    """
    if not datasets:
        raise ValueError("No datasets provided")
    
    # Standardize each dataset
    processed_datasets = []
    for df in datasets:
        df = normalize_column_names(df)
        df = standardize_churn_column(df)
        processed_datasets.append(df)
    
    # Concatenate datasets
    merged_df = pd.concat(processed_datasets, axis=0, ignore_index=True)
    
    print(f"\nMerged dataset shape: {merged_df.shape}")
    print(f"Columns: {list(merged_df.columns)}")
    print(f"\nChurn distribution:")
    print(merged_df['Churn'].value_counts())
    
    return merged_df


def log_data_version(merged_df: pd.DataFrame, experiments_path: str):
    """
    Log data version information for versioning.
    
    Args:
        merged_df: Merged DataFrame
        experiments_path: Path to experiments directory
    """
    version_info = {
        'timestamp': datetime.now().isoformat(),
        'shape': merged_df.shape,
        'rows': int(merged_df.shape[0]),
        'columns': int(merged_df.shape[1]),
        'churn_distribution': merged_df['Churn'].value_counts().to_dict(),
        'missing_values': merged_df.isnull().sum().to_dict()
    }
    
    version_file = os.path.join(experiments_path, 'data_version.json')
    with open(version_file, 'w') as f:
        json.dump(version_info, f, indent=4)
    
    print(f"\nData version logged to {version_file}")
    return version_info


if __name__ == "__main__":
    # Example usage
    raw_path = "data/raw"
    processed_path = "data/processed"
    experiments_path = "experiments"
    
    datasets = load_all_datasets(raw_path)
    merged = merge_datasets(datasets)
    
    merged.to_csv(os.path.join(processed_path, 'merged_dataset.csv'), index=False)
    log_data_version(merged, experiments_path)
    
    print("\nMerge completed successfully!")
