"""Feature processing and engineering module."""
import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime, timedelta
from ..config.aml_config import AML_CONFIG, FEATURE_COLUMNS
import logging

logger = logging.getLogger(__name__)

class FeatureProcessor:
    def __init__(self):
        self.aml_config = AML_CONFIG
    
    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all features for AML detection."""
        try:
            # Validate required columns
            self._validate_columns(df)
            
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            # Normalize data
            df = self._normalize_data(df)
            
            # Add features
            df = self._add_transaction_features(df)
            df = self._add_country_risk_features(df)
            df = self._add_structuring_features(df)
            
            # Detect suspicious transactions
            df = self._detect_suspicious_transactions(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing features: {str(e)}")
            raise
    
    def _validate_columns(self, df: pd.DataFrame):
        """Validate required columns exist."""
        missing_cols = FEATURE_COLUMNS['required'] - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize dates and currencies."""
        df = self._normalize_dates(df)
        df = self._normalize_amounts(df)
        return df
    
    def _normalize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert dates to datetime format."""
        try:
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            return df
        except Exception as e:
            logger.error(f"Error normalizing dates: {str(e)}")
            raise
    
    def _normalize_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize transaction amounts."""
        try:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df['amount'] = df['amount'].fillna(0)
            return df
        except Exception as e:
            logger.error(f"Error normalizing amounts: {str(e)}")
            raise
    
    def _add_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add transaction-based features."""
        try:
            # Daily transaction count per sender
            daily_counts = df.groupby(['sender_id', 'transaction_date']).size()
            df['daily_tx_count'] = df.groupby('sender_id')['transaction_date'].transform('count')
            
            # Daily total amount per sender
            df['daily_total_amount'] = df.groupby(['sender_id', 'transaction_date'])['amount'].transform('sum')
            
            # Daily average transaction size
            df['daily_avg_amount'] = df.groupby(['sender_id', 'transaction_date'])['amount'].transform('mean')
            
            # Rolling statistics (7-day window)
            df = df.sort_values('transaction_date')
            
            # Calculate rolling statistics using time-based resampling
            def calculate_rolling_stats(group):
                # Resample to daily frequency and calculate rolling stats
                daily = group.resample('D', on='transaction_date')['amount'].agg(['mean', 'std']).fillna(method='ffill')
                daily['mean'] = daily['mean'].rolling(window=7, min_periods=1).mean()
                daily['std'] = daily['std'].rolling(window=7, min_periods=1).std()
                # Map back to original dates
                group['rolling_avg'] = group['transaction_date'].map(daily['mean'])
                group['rolling_std'] = group['transaction_date'].map(daily['std'])
                return group

            # Apply rolling calculations per sender
            df = df.groupby('sender_id', group_keys=False).apply(calculate_rolling_stats)
            
            # Fill NaN values with appropriate defaults
            df['rolling_avg_amount'] = df['rolling_avg'].fillna(df['amount'])
            df['rolling_std_amount'] = df['rolling_std'].fillna(0)
            
            # Clean up temporary columns
            df = df.drop(['rolling_avg', 'rolling_std'], axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding transaction features: {str(e)}")
            raise
    
    def _add_country_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add country risk-based features."""
        try:
            # Flag high-risk countries
            high_risk_countries = self.aml_config['high_risk_countries']
            df['sender_high_risk'] = df['sender_country'].isin(high_risk_countries)
            df['recipient_high_risk'] = df['recipient_country'].isin(high_risk_countries)
            
            # Cross-border transaction flag
            df['is_cross_border'] = df['sender_country'] != df['recipient_country']
            
            # Country risk scores (simplified example)
            country_risk_scores = {
                country: 1.0 if country in high_risk_countries else 0.2
                for country in set(df['sender_country'].unique()) | set(df['recipient_country'].unique())
            }
            
            df['sender_country_risk_score'] = df['sender_country'].map(country_risk_scores)
            df['recipient_country_risk_score'] = df['recipient_country'].map(country_risk_scores)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding country risk features: {str(e)}")
            raise
    
    def _add_structuring_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features to detect potential structuring."""
        try:
            threshold = self.aml_config['structuring_threshold']
            window_days = self.aml_config['structuring_window_days']
            
            # Sort by date for window calculations
            df = df.sort_values('transaction_date')
            
            def detect_structuring(group):
                """Detect potential structuring in a group of transactions."""
                if len(group) < 3:
                    return pd.Series([False] * len(group))
                
                # Calculate rolling statistics using fixed window
                rolling_mean = group['amount'].rolling(window=3, min_periods=3).mean()
                rolling_std = group['amount'].rolling(window=3, min_periods=3).std()
                
                # Detect potential structuring patterns
                is_structured = (
                    (rolling_mean > threshold * 0.7) &
                    (rolling_mean < threshold * 0.95) &
                    (rolling_std / rolling_mean < 0.2)
                )
                
                return is_structured.fillna(False)
            
            # Apply structuring detection per sender within time windows
            df['potential_structuring'] = df.groupby('sender_id', group_keys=False).apply(
                lambda x: detect_structuring(x)
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding structuring features: {str(e)}")
            raise
    
    def _detect_suspicious_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect suspicious transactions based on multiple criteria."""
        try:
            # Initialize suspicious flag
            df['is_suspicious'] = False
            df['suspicious_reasons'] = ''
            
            # Rule 1: Large transactions
            large_tx = df['amount'] > self.aml_config['transaction_threshold']
            df.loc[large_tx, 'suspicious_reasons'] += 'large_transaction;'
            
            # Rule 2: High frequency transactions
            high_freq = df['daily_tx_count'] > self.aml_config['high_frequency_threshold']
            df.loc[high_freq, 'suspicious_reasons'] += 'high_frequency;'
            
            # Rule 3: Unusual amount (outside 3 standard deviations)
            unusual_amount = (
                (df['amount'] > df['rolling_avg_amount'] + 3 * df['rolling_std_amount']) &
                (df['rolling_std_amount'] > 0)  # Only where we have enough data
            )
            df.loc[unusual_amount, 'suspicious_reasons'] += 'unusual_amount;'
            
            # Rule 4: Structuring patterns
            df.loc[df['potential_structuring'], 'suspicious_reasons'] += 'potential_structuring;'
            
            # Rule 5: High-risk country involvement
            high_risk = df['sender_high_risk'] | df['recipient_high_risk']
            df.loc[high_risk, 'suspicious_reasons'] += 'high_risk_country;'
            
            # Rule 6: Cross-border high-value transactions
            cross_border_high_value = (
                df['is_cross_border'] & 
                (df['amount'] > self.aml_config['transaction_threshold'] * 0.8)
            )
            df.loc[cross_border_high_value, 'suspicious_reasons'] += 'cross_border_high_value;'
            
            # Mark as suspicious if any rules were triggered
            df['is_suspicious'] = df['suspicious_reasons'] != ''
            
            # Calculate risk score (0-1)
            df['risk_score'] = 0.0
            
            # Weight different factors
            weights = {
                'large_transaction': 0.3,
                'high_frequency': 0.15,
                'unusual_amount': 0.2,
                'potential_structuring': 0.25,
                'high_risk_country': 0.25,
                'cross_border_high_value': 0.2
            }
            
            for reason, weight in weights.items():
                df.loc[df['suspicious_reasons'].str.contains(reason, na=False), 'risk_score'] += weight
            
            # Normalize risk score to 0-1
            df['risk_score'] = df['risk_score'].clip(0, 1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error detecting suspicious transactions: {str(e)}")
            raise 