"""
Feature engineering module for creating advanced features.
"""
import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Handles feature engineering and selection.
    """
    
    def __init__(self):
        self.feature_importance = {}
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new engineered features from existing ones.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target if present
        if 'Churn' in numeric_cols:
            numeric_cols.remove('Churn')
        
        # Feature interactions for top numeric features
        if len(numeric_cols) >= 2:
            # Create interaction features (limited to avoid explosion)
            for i, col1 in enumerate(numeric_cols[:3]):
                for col2 in numeric_cols[i+1:4]:
                    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            
            # Create polynomial features for top columns
            for col in numeric_cols[:2]:
                if col in df.columns:
                    df[f'{col}_squared'] = df[col] ** 2
        
        # Domain-specific features (if certain columns exist)
        if 'tenure' in df.columns:
            df['tenure_squared'] = df['tenure'] ** 2
            df['tenure_binned'] = pd.cut(df['tenure'], bins=5, labels=False)
        
        if 'monthlycharges' in df.columns and 'tenure' in df.columns:
            df['customer_lifetime_value'] = df['monthlycharges'] * df['tenure']
        
        # Log transformations for skewed features
        for col in numeric_cols:
            if col in df.columns and (df[col] > 0).all():
                df[f'{col}_log'] = np.log1p(df[col])
        
        print(f"Original features: {len(numeric_cols)}")
        print(f"New features created: {df.shape[1] - len(numeric_cols) - 1}")
        print(f"Total features: {df.shape[1] - 1}")
        
        return df
    
    def select_top_features(self, df: pd.DataFrame, target_col: str = 'Churn', 
                           top_n: int = 20) -> list:
        """
        Select top features based on correlation with target.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            top_n: Number of top features to select
            
        Returns:
            List of top feature names
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col not in df.columns or target_col not in numeric_cols:
            print("Target column not found or not numeric")
            return numeric_cols[:top_n]
        
        # Calculate correlation with target
        correlations = df[numeric_cols].corr()[target_col].abs()
        correlations = correlations.drop(target_col, errors='ignore')
        
        # Sort by absolute correlation
        top_features = correlations.nlargest(top_n).index.tolist()
        
        self.feature_importance = correlations.to_dict()
        
        print(f"\nTop {top_n} features by correlation with {target_col}:")
        for i, (feature, corr) in enumerate(correlations.nlargest(top_n).items(), 1):
            print(f"{i}. {feature}: {corr:.4f}")
        
        return top_features


def feature_engineering_pipeline(input_path: str, output_path: str,
                                target_col: str = 'Churn') -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        input_path: Path to preprocessed data
        output_path: Path to save engineered features
        target_col: Target column name
        
    Returns:
        DataFrame with engineered features
    """
    # Load data
    df = pd.read_csv(input_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Engineer features
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(df)
    
    # Select top features
    top_features = engineer.select_top_features(df_engineered, target_col)
    
    # Keep target and top features
    keep_cols = [target_col] + top_features
    keep_cols = [col for col in keep_cols if col in df_engineered.columns]
    
    df_final = df_engineered[keep_cols]
    
    # Save engineered features
    df_final.to_csv(output_path, index=False)
    print(f"\nSaved engineered features to {output_path}")
    print(f"Final shape: {df_final.shape}")
    
    return df_final


if __name__ == "__main__":
    input_path = "data/processed/preprocessed_data.csv"
    output_path = "data/processed/engineered_features.csv"
    
    df = feature_engineering_pipeline(input_path, output_path)
    print("\nFeature engineering completed successfully!")
