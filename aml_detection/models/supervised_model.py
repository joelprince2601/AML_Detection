"""Supervised learning module for suspicious transaction detection."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from ..utils.logging_utils import logger
from ..config.aml_config import FEATURE_COLUMNS

class SuspiciousTransactionClassifier:
    def __init__(self, test_size: float = 0.2):
        self.test_size = test_size
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, df: pd.DataFrame) -> Tuple[RandomForestClassifier, Dict]:
        """
        Train the model and return performance metrics.
        
        Args:
            df (pd.DataFrame): DataFrame with features and 'is_suspicious' label
            
        Returns:
            Tuple[RandomForestClassifier, Dict]: Trained model and performance metrics
        """
        features = FEATURE_COLUMNS['numeric']
        
        # Ensure all required features are present
        missing_features = set(features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        if 'is_suspicious' not in df.columns:
            raise ValueError("Target column 'is_suspicious' not found in DataFrame")
        
        X = df[features]
        y = df['is_suspicious']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Generate performance metrics
        metrics = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': feature_importance,
            'accuracy': self.model.score(X_test, y_test)
        }
        
        logger.info(f"Model training completed. Accuracy: {metrics['accuracy']:.3f}")
        logger.info("\nClassification Report:\n" + metrics['classification_report'])
        
        return self.model, metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            df (pd.DataFrame): DataFrame with features
            
        Returns:
            pd.DataFrame: DataFrame with predictions added
        """
        features = FEATURE_COLUMNS['numeric']
        
        # Ensure all required features are present
        missing_features = set(features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Handle missing values
        X = df[features].fillna(df[features].mean())
        
        # Make predictions
        df['predicted_suspicious'] = self.model.predict(X)
        df['suspicious_probability'] = self.model.predict_proba(X)[:, 1]
        
        return df 