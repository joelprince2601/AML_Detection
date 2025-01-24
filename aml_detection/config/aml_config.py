"""Configuration settings for AML detection."""

AML_CONFIG = {
    'high_risk_countries': {
        'NK', 'IR', 'MM', 'CU', 'SY'  # Example: North Korea, Iran, Myanmar, Cuba, Syria
    },
    'transaction_threshold': 10000,  # Standard reporting threshold
    'structuring_threshold': 9000,  # Slightly below reporting threshold
    'structuring_window_days': 7,  # Time window to check for structuring
    'high_frequency_threshold': 10  # Number of transactions per day considered high
}

# Feature configurations
FEATURE_COLUMNS = {
    'required': {'transaction_date', 'amount', 'sender_country', 'recipient_country'},
    'numeric': [
        'amount', 'daily_tx_count', 'daily_total_amount',
        'daily_avg_amount', 'rolling_avg_amount', 'rolling_std_amount',
        'sender_country_risk_score', 'recipient_country_risk_score'
    ]
} 