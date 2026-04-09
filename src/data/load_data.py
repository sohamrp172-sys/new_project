"""
Improved data loading module - uses single high-quality dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import json
import os
from datetime import datetime

from src.utils.config import config
from src.utils.logger import logger


def load_primary_dataset(dataset_path: Optional[str] = None, 
                        sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load primary dataset (single high-quality source).
    
    Args:
        dataset_path: Path to dataset file
        sample_size: Number of rows to sample (None for full dataset)
        
    Returns:
        Loaded DataFrame
    """
    if dataset_path is None:
        dataset_path = config.get('data.primary_dataset')
    
    if sample_size is None:
        sample_size = config.get('data.sample_size')
    
    logger.info(f"Loading dataset from: {dataset_path}")
    
    file_path = Path(dataset_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Load based on file extension
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Loaded dataset shape: {df.shape}")
    
    # Sample if requested
    if sample_size and len(df) > sample_size:
        logger.info(f"Sampling {sample_size} rows from {len(df)} total")
        df = df.sample(n=sample_size, random_state=config.get('data.random_state', 42))
        df = df.reset_index(drop=True)
    
    return df


def standardize_churn_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the Churn column to binary (0, 1).
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized Churn column
    """
    df = df.copy()
    
    # Find churn column (case-insensitive)
    churn_cols = [col for col in df.columns if 'churn' in col.lower()]
    
    if not churn_cols:
        logger.warning("No churn column found. Skipping standardization.")
        return df
    
    churn_col = churn_cols[0]
    logger.info(f"Found churn column: {churn_col}")
    
    # Convert to string and standardize
    df[churn_col] = df[churn_col].astype(str).str.lower().str.strip()
    
    # Map to binary
    churn_mapping = {
        'yes': 1, 'no': 0,
        '1': 1, '0': 0,
        'true': 1, 'false': 0,
        '1.0': 1, '0.0': 0,
        'y': 1, 'n': 0
    }
    
    df[churn_col] = df[churn_col].map(churn_mapping)
    
    # Rename to standard name
    if churn_col != 'Churn':
        df.rename(columns={churn_col: 'Churn'}, inplace=True)
    
    # Log distribution
    churn_dist = df['Churn'].value_counts()
    logger.info(f"Churn distribution:\n{churn_dist}")
    
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize column names.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned column names
    """
    df = df.copy()
    
    # Store churn column if it exists
    churn_col = None
    for col in df.columns:
        if 'churn' in col.lower():
            churn_col = col
            break
    
    # Remove whitespace, lowercase, replace spaces with underscores
    df.columns = (df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(' ', '_')
                  .str.replace('-', '_')
                  .str.replace('.', '_'))
    
    return df


def remove_sparse_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Remove columns with too many missing values.
    
    Args:
        df: Input DataFrame
        threshold: Maximum fraction of missing values allowed
        
    Returns:
        DataFrame with sparse columns removed
    """
    df = df.copy()
    
    missing_pct = df.isnull().sum() / len(df)
    sparse_cols = missing_pct[missing_pct > threshold].index.tolist()
    
    if sparse_cols:
        logger.info(f"Removing {len(sparse_cols)} sparse columns (>{threshold*100}% missing): {sparse_cols}")
        df = df.drop(columns=sparse_cols)
    
    return df


def log_data_version(df: pd.DataFrame, experiments_path: str = "experiments"):
    """
    Log data version information.
    
    Args:
        df: DataFrame to log
        experiments_path: Path to experiments directory
    """
    version_info = {
        'timestamp': datetime.now().isoformat(),
        'shape': list(df.shape),
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'column_names': list(df.columns),
        'churn_distribution': df['Churn'].value_counts().to_dict() if 'Churn' in df.columns else {},
        'missing_values_per_column': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    
    os.makedirs(experiments_path, exist_ok=True)
    version_file = os.path.join(experiments_path, 'data_version.json')
    
    with open(version_file, 'w') as f:
        json.dump(version_info, f, indent=4)
    
    logger.info(f"Data version logged to {version_file}")


def load_and_prepare_data(dataset_path: Optional[str] = None,
                         sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Complete data loading and preparation pipeline.
    
    Args:
        dataset_path: Path to dataset
        sample_size: Sample size
        
    Returns:
        Prepared DataFrame
    """
    logger.info("="*70)
    logger.info("DATA LOADING AND PREPARATION")
    logger.info("="*70)
    
    # Load data
    df = load_primary_dataset(dataset_path, sample_size)
    
    # Standardize churn column FIRST (before cleaning names)
    df = standardize_churn_column(df)
    
    # Clean column names
    df = clean_column_names(df)
    
    # Remove sparse columns (but keep Churn)
    threshold = config.get('preprocessing.missing_value_threshold', 0.5)
    churn_col = df['churn'] if 'churn' in df.columns else None
    df = remove_sparse_columns(df, threshold)
    if churn_col is not None and 'churn' not in df.columns:
        df['churn'] = churn_col
    
    # Log version
    log_data_version(df)
    
    logger.info(f"Final prepared data shape: {df.shape}")
    logger.info("Data loading completed successfully")
    
    return df


if __name__ == "__main__":
    df = load_and_prepare_data()
    print(f"\nLoaded data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
