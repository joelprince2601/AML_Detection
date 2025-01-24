"""Create example transaction dataset."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

def create_sample_data(n_transactions=1000):
    """Create sample transaction data."""
    # Create dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(n_transactions)]
    
    # Create sender and recipient IDs
    n_accounts = 100
    sender_ids = [f"S{i:03d}" for i in range(n_accounts)]
    recipient_ids = [f"R{i:03d}" for i in range(n_accounts)]
    
    # Create countries
    countries = ['US', 'UK', 'FR', 'DE', 'JP', 'CN', 'IR', 'NK']  # Including high-risk countries
    
    # Generate data
    df = pd.DataFrame({
        'transaction_date': np.random.choice(dates, n_transactions),
        'amount': np.random.lognormal(mean=8, sigma=1, size=n_transactions),
        'sender_id': np.random.choice(sender_ids, n_transactions),
        'sender_country': np.random.choice(countries, n_transactions, p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.03, 0.02]),
        'recipient_id': np.random.choice(recipient_ids, n_transactions),
        'recipient_country': np.random.choice(countries, n_transactions, p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.03, 0.02])
    })
    
    # Sort by date
    df = df.sort_values('transaction_date')
    
    return df

if __name__ == "__main__":
    # Create example data
    df = create_sample_data()
    
    # Create example data directory
    example_dir = Path(__file__).parent / "aml_detection" / "example_data"
    example_dir.mkdir(parents=True, exist_ok=True)
    
    # Save example data
    df.to_excel(example_dir / "sample_transactions.xlsx", index=False) 