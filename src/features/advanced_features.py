"""
Advanced feature engineering with domain knowledge and better selection.
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from typing import List, Optional
import json, os

from src.utils.config import config
from src.utils.logger import logger


class AdvancedFeatureEngineer:
    """Advanced feature engineering with domain knowledge."""

    def __init__(self):
        self.feature_importance: dict = {}
        self.selected_features: List[str] = []
        # Store training-time statistics to avoid data leakage
        self._monthly_charges_p75: Optional[float] = None

    # ------------------------------------------------------------------
    # Domain features
    # ------------------------------------------------------------------
    def create_domain_features(self, df: pd.DataFrame,
                               is_training: bool = True) -> pd.DataFrame:
        """
        Create domain-specific features for telecom churn.

        Args:
            df: Input DataFrame
            is_training: If True, compute and store thresholds from this data.
                         If False, use stored thresholds (avoids data leakage).
        """
        df = df.copy()
        logger.info("Creating domain-specific features...")

        # Tenure-based
        if 'tenure' in df.columns:
            df['is_new_customer']      = (df['tenure'] < 6).astype(int)
            df['is_long_term_customer'] = (df['tenure'] > 36).astype(int)
            df['tenure_squared']       = df['tenure'] ** 2
            df['tenure_log']           = np.log1p(df['tenure'].clip(lower=0))

        # Revenue-based
        if 'monthlycharges' in df.columns:
            df['monthlycharges_log'] = np.log1p(df['monthlycharges'].clip(lower=0))

            # FIX: compute percentile on training data only, reuse at inference
            if is_training:
                self._monthly_charges_p75 = float(df['monthlycharges'].quantile(0.75))
                logger.info(f"Stored monthlycharges p75 = {self._monthly_charges_p75:.2f}")
            threshold = self._monthly_charges_p75 if self._monthly_charges_p75 is not None else 75.0
            df['is_high_value'] = (df['monthlycharges'] > threshold).astype(int)

        if 'totalcharges' in df.columns:
            df['totalcharges_log'] = np.log1p(df['totalcharges'].clip(lower=0))

        # Customer lifetime value
        if 'monthlycharges' in df.columns and 'tenure' in df.columns:
            df['customer_lifetime_value'] = df['monthlycharges'] * df['tenure']
            df['avg_monthly_spend']       = df['totalcharges'] / (df['tenure'] + 1)
            df['tenure_to_monthly_ratio'] = df['tenure'] / (df['monthlycharges'] + 1)

        # Contract
        if 'contract' in df.columns:
            df['has_long_contract'] = (df['contract'] > 0).astype(int)

        # Service usage
        if 'num_services' in df.columns:
            df['service_usage_low']  = (df['num_services'] <= 2).astype(int)
            df['service_usage_high'] = (df['num_services'] >= 5).astype(int)

        # Payment behaviour
        if 'paperless_billing' in df.columns and 'payment_method' in df.columns:
            df['digital_customer'] = (
                (df['paperless_billing'] == 1) & (df['payment_method'] != 0)
            ).astype(int)

        # Complaints / support
        if 'num_complaints' in df.columns:
            df['has_complaints']  = (df['num_complaints'] > 0).astype(int)
            df['high_complaints'] = (df['num_complaints'] > 2).astype(int)

        if 'num_service_calls' in df.columns:
            df['frequent_support_user'] = (df['num_service_calls'] > 3).astype(int)

        logger.info(f"Domain features created, shape now: {df.shape}")
        return df

    # ------------------------------------------------------------------
    # Interaction features  (FIX: select top features FIRST, then interact)
    # ------------------------------------------------------------------
    def create_interaction_features(self, df: pd.DataFrame,
                                    top_n: int = 5) -> pd.DataFrame:
        """Create pairwise interactions between the top_n numeric columns."""
        df = df.copy()
        logger.info("Creating interaction features...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for drop_col in ('Churn', 'churn'):
            if drop_col in numeric_cols:
                numeric_cols.remove(drop_col)

        # FIX: exclude ID-like and date columns — they produce meaningless interactions
        exclude_patterns = ('id', 'date', 'customerid', 'customer_id', 'signup')
        numeric_cols = [
            c for c in numeric_cols
            if not any(pat in c.lower() for pat in exclude_patterns)
        ]

        # Limit BEFORE creating interactions to avoid explosion
        numeric_cols = numeric_cols[:top_n]

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                # Safe ratio — return 0 when denominator is near zero
                df[f'{col1}_div_{col2}'] = np.where(
                    np.abs(df[col2]) > 1e-3,
                    df[col1] / df[col2],
                    0.0
                )

        logger.info(f"Interaction features created, shape now: {df.shape}")
        return df

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------
    def select_features_tree_importance(self, X: pd.DataFrame, y: pd.Series,
                                        n_features: int = 20) -> List[str]:
        """Select top features by Random Forest importance."""
        logger.info("Selecting features using tree importance...")

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            random_state=config.get('data.random_state', 42),
            n_jobs=-1
        )
        rf.fit(X, y)

        importances = pd.Series(rf.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)

        selected = importances.head(n_features).index.tolist()
        self.feature_importance = importances.to_dict()
        self.selected_features  = selected

        logger.info(f"Top {min(10, len(selected))} features by importance:")
        for i, (feat, imp) in enumerate(importances.head(10).items(), 1):
            logger.info(f"  {i}. {feat}: {imp:.4f}")

        return selected

    def select_features_mutual_info(self, X: pd.DataFrame, y: pd.Series,
                                    n_features: int = 20) -> List[str]:
        """Select top features by mutual information."""
        logger.info("Selecting features using mutual information...")

        selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))
        selector.fit(X, y)

        scores   = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
        selected = scores.head(n_features).index.tolist()
        self.feature_importance = scores.to_dict()
        self.selected_features  = selected

        logger.info(f"Top {min(10, len(selected))} features by mutual info:")
        for i, (feat, score) in enumerate(scores.head(10).items(), 1):
            logger.info(f"  {i}. {feat}: {score:.4f}")

        return selected

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def engineer_and_select(self, X: pd.DataFrame, y: pd.Series,
                             is_training: bool = True) -> pd.DataFrame:
        """Complete feature engineering + selection pipeline."""
        logger.info("=" * 70)
        logger.info("FEATURE ENGINEERING AND SELECTION")
        logger.info("=" * 70)

        if config.get('feature_engineering.create_domain_features', True):
            X = self.create_domain_features(X, is_training=is_training)

        if config.get('feature_engineering.create_interactions', True):
            X = self.create_interaction_features(X, top_n=5)

        logger.info(f"Features after engineering: {X.shape[1]}")

        # Clean up NaN / inf
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        n_features       = config.get('feature_engineering.n_features', 20)
        selection_method = config.get('feature_engineering.selection_method', 'tree_importance')

        if selection_method == 'tree_importance':
            selected = self.select_features_tree_importance(X, y, n_features)
        elif selection_method == 'mutual_info':
            selected = self.select_features_mutual_info(X, y, n_features)
        else:
            selected = X.columns.tolist()

        X_selected = X[selected]
        logger.info(f"Final feature count: {X_selected.shape[1]}")
        logger.info("Feature engineering completed")
        return X_selected

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_state(self, path: str):
        """Save thresholds and selected features so inference is consistent."""
        state = {
            'monthly_charges_p75': self._monthly_charges_p75,
            'selected_features':   self.selected_features,
            'feature_importance':  self.feature_importance,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=4)
        logger.info(f"Feature engineer state saved to {path}")

    def load_state(self, path: str):
        """Load thresholds from a previous training run."""
        with open(path) as f:
            state = json.load(f)
        self._monthly_charges_p75 = state.get('monthly_charges_p75')
        self.selected_features    = state.get('selected_features', [])
        self.feature_importance   = state.get('feature_importance', {})
        logger.info(f"Feature engineer state loaded from {path}")
