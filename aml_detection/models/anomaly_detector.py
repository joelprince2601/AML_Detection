"""Anomaly detection module."""
from sklearn.ensemble import IsolationForest
import pandas as pd
from ..config.aml_config import FEATURE_COLUMNS
from ..utils.logging_utils import logger

class AnomalyDetector:
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in transaction data."""
        features = FEATURE_COLUMNS['numeric']
        
        df['is_anomaly'] = self.model.fit_predict(df[features])
        df['is_anomaly'] = df['is_anomaly'].map({1: 0, -1: 1})
        df['anomaly_score'] = -self.model.score_samples(df[features])
        
        logger.info(f"Detected {df['is_anomaly'].sum()} anomalies")
        return df 