"""
Data preprocessing module for cleaning and preparing data.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import json

from src.utils.logger import logger


class DataPreprocessor:
    """Handles data cleaning, encoding, and scaling."""

    def __init__(self):
        self.scaler          = StandardScaler()
        self.encoders: dict  = {}
        self.numeric_cols:   list = []
        self.categorical_cols: list = []
        self.feature_names:  list = []

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, target_col: str = 'Churn') -> pd.DataFrame:
        df = df.copy()

        self.numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        for col_list in (self.numeric_cols, self.categorical_cols):
            if target_col and target_col in col_list:
                col_list.remove(target_col)

        df = self._handle_missing_values(df)
        df = self._encode_categorical(df, fit=True)
        df = self._scale_numeric(df, fit=True)

        self.feature_names = [c for c in df.columns if c != target_col]
        return df

    def transform(self, df: pd.DataFrame, target_col: str = 'Churn') -> pd.DataFrame:
        df = df.copy()
        df = self._handle_missing_values(df)
        df = self._encode_categorical(df, fit=False)
        df = self._scale_numeric(df, fit=False)
        return df

    def fit_transform(self, df: pd.DataFrame, target_col: str = 'Churn') -> pd.DataFrame:
        return self.fit(df, target_col)

    # ------------------------------------------------------------------
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        before = int(df.isnull().sum().sum())
        logger.info(f"Missing values before: {before}")

        for col in self.numeric_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        for col in self.categorical_cols:
            if col in df.columns and df[col].isnull().any():
                mode_vals = df[col].mode()
                if len(mode_vals):
                    df[col] = df[col].fillna(mode_vals[0])

        logger.info(f"Missing values after:  {int(df.isnull().sum().sum())}")
        return df

    def _encode_categorical(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            if fit:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.encoders:
                    # Handle unseen labels gracefully
                    known = set(self.encoders[col].classes_)
                    df[col] = df[col].astype(str).apply(
                        lambda v: v if v in known else self.encoders[col].classes_[0]
                    )
                    df[col] = self.encoders[col].transform(df[col])
        return df

    def _scale_numeric(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        cols = [c for c in self.numeric_cols if c in df.columns]
        if not cols:
            return df
        if fit:
            df[cols] = self.scaler.fit_transform(df[cols])
        else:
            df[cols] = self.scaler.transform(df[cols])
        return df

    # ------------------------------------------------------------------
    # Persistence — save/load encoders and scaler so inference is reproducible
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save preprocessor state (scaler + encoders) to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'scaler':          self.scaler,
            'encoders':        self.encoders,
            'numeric_cols':    self.numeric_cols,
            'categorical_cols': self.categorical_cols,
            'feature_names':   self.feature_names,
        }
        joblib.dump(state, path)
        logger.info(f"Preprocessor saved to {path}")

    def load(self, path: str):
        """Load preprocessor state from disk."""
        state = joblib.load(path)
        self.scaler           = state['scaler']
        self.encoders         = state['encoders']
        self.numeric_cols     = state['numeric_cols']
        self.categorical_cols = state['categorical_cols']
        self.feature_names    = state['feature_names']
        logger.info(f"Preprocessor loaded from {path}")
