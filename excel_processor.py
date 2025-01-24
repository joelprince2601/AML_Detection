import pandas as pd
import numpy as np
from datetime import datetime
import gc
from typing import Optional, List, Dict, Set, Tuple
import logging
import json
from pathlib import Path
import networkx as nx

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExcelProcessor:
    def __init__(self, file_path: str):
        """
        Initialize the Excel processor with the file path and AML configs.
        
        Args:
            file_path (str): Path to the Excel file
        """
        self.file_path = file_path
        self.dtype_dict = None  # Will store optimal data types
        
        # Load AML configuration
        self.aml_config = {
            'high_risk_countries': {
                'NK', 'IR', 'MM', 'CU', 'SY'  # Example: North Korea, Iran, Myanmar, Cuba, Syria
            },
            'transaction_threshold': 10000,  # Standard reporting threshold
            'structuring_threshold': 9000,  # Slightly below reporting threshold
            'structuring_window_days': 7,  # Time window to check for structuring
            'high_frequency_threshold': 10  # Number of transactions per day considered high
        }
        
    def read_in_chunks(self, chunk_size: int = 100000) -> pd.DataFrame:
        """
        Read large Excel files in chunks to manage memory efficiently.
        
        Args:
            chunk_size (int): Number of rows to read in each chunk
            
        Yields:
            pd.DataFrame: Processed chunk of data
        """
        try:
            # First pass: determine optimal dtypes
            logger.info(f"Reading first chunk to determine data types...")
            first_chunk = pd.read_excel(
                self.file_path,
                nrows=chunk_size,
                engine='openpyxl'
            )
            
            # Optimize dtypes for numeric and categorical columns
            self.dtype_dict = self._optimize_dtypes(first_chunk)
            
            # Read the actual file in chunks with optimized dtypes
            for chunk in pd.read_excel(
                self.file_path,
                chunksize=chunk_size,
                dtype=self.dtype_dict,
                engine='openpyxl'
            ):
                # Process the chunk
                processed_chunk = self._process_chunk(chunk)
                yield processed_chunk
                
                # Force garbage collection
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            raise
            
    def _optimize_dtypes(self, df: pd.DataFrame) -> Dict:
        """
        Determine optimal data types for each column.
        
        Args:
            df (pd.DataFrame): Sample dataframe to analyze
            
        Returns:
            Dict: Mapping of column names to optimal data types
        """
        dtypes = {}
        
        for column in df.columns:
            # Skip date columns
            if df[column].dtype == 'datetime64[ns]':
                continue
                
            # Try to convert to categorical for string columns with low cardinality
            if df[column].dtype == 'object':
                unique_count = df[column].nunique()
                if unique_count / len(df) < 0.5:  # If less than 50% unique values
                    dtypes[column] = 'category'
                    
            # Optimize numeric columns
            elif pd.api.types.is_numeric_dtype(df[column]):
                if df[column].isnull().any():
                    dtypes[column] = 'float32'  # Use float32 if nulls present
                else:
                    dtypes[column] = 'int32' if df[column].dtype == 'int64' else 'float32'
                    
        return dtypes
    
    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Process each chunk of data: handle missing values, normalize dates and currencies.
        
        Args:
            chunk (pd.DataFrame): DataFrame chunk to process
            
        Returns:
            pd.DataFrame: Processed DataFrame chunk
        """
        # Handle missing values
        chunk = self._handle_missing_values(chunk)
        
        # Normalize dates
        chunk = self._normalize_dates(chunk)
        
        # Normalize currencies
        chunk = self._normalize_currencies(chunk)
        
        return chunk
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on data type."""
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column].fillna(df[column].mean(), inplace=True)
            elif pd.api.types.is_datetime64_dtype(df[column]):
                df[column].fillna(pd.NaT, inplace=True)
            else:
                df[column].fillna('MISSING', inplace=True)
        return df
    
    def _normalize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert date columns to datetime format."""
        for column in df.columns:
            if pd.api.types.is_datetime64_dtype(df[column]):
                df[column] = pd.to_datetime(df[column], errors='coerce')
        return df
    
    def _normalize_currencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize currency columns to float values."""
        currency_columns = [col for col in df.columns if df[col].dtype == 'object' and 
                          df[col].str.contains(r'[$£€¥]', na=False).any()]
        
        for column in currency_columns:
            df[column] = df[column].replace('[\$,£,€,¥]', '', regex=True)
            df[column] = pd.to_numeric(df[column], errors='coerce')
        return df

    def add_aml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add AML-related features to the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with transaction data
            
        Returns:
            pd.DataFrame: DataFrame with additional AML features
        """
        # Ensure required columns exist
        required_cols = {'transaction_date', 'amount', 'sender_country', 'recipient_country'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns. Expected: {required_cols}")

        # Convert date column to datetime if not already
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # Add basic transaction features
        df = self._add_transaction_features(df)
        
        # Add country risk features
        df = self._add_country_risk_features(df)
        
        # Add structuring detection features
        df = self._add_structuring_features(df)
        
        return df
    
    def _add_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add transaction-related features."""
        # Group by sender and date to calculate daily metrics
        daily_stats = df.groupby([
            'sender_id',
            pd.Grouper(key='transaction_date', freq='D')
        ]).agg({
            'amount': ['count', 'sum', 'mean', 'std'],
            'transaction_id': 'count'
        }).reset_index()
        
        daily_stats.columns = [
            'sender_id', 'date', 'daily_tx_count', 
            'daily_total_amount', 'daily_avg_amount', 'daily_std_amount'
        ]
        
        # Calculate rolling averages (30-day window)
        rolling_stats = df.groupby('sender_id').rolling('30D', on='transaction_date').agg({
            'amount': ['mean', 'std']
        }).reset_index()
        
        rolling_stats.columns = ['sender_id', 'rolling_avg_amount', 'rolling_std_amount']
        
        # Merge features back to original DataFrame
        df = df.merge(daily_stats, on=['sender_id', 'transaction_date'], how='left')
        df = df.merge(rolling_stats, on='sender_id', how='left')
        
        # Flag high-frequency trading
        df['is_high_frequency'] = df['daily_tx_count'] > self.aml_config['high_frequency_threshold']
        
        return df
    
    def _add_country_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add country risk-related features."""
        # Flag transactions involving high-risk countries
        df['sender_high_risk'] = df['sender_country'].isin(self.aml_config['high_risk_countries'])
        df['recipient_high_risk'] = df['recipient_country'].isin(self.aml_config['high_risk_countries'])
        df['is_cross_border'] = df['sender_country'] != df['recipient_country']
        
        # Calculate country risk score (example method)
        country_risk_scores = self._calculate_country_risk_scores(df)
        df['sender_country_risk_score'] = df['sender_country'].map(country_risk_scores)
        df['recipient_country_risk_score'] = df['recipient_country'].map(country_risk_scores)
        
        return df
    
    def _add_structuring_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features to detect potential structuring behavior."""
        window = f"{self.aml_config['structuring_window_days']}D"
        threshold = self.aml_config['structuring_threshold']
        
        # Find transactions just below threshold
        df['is_below_threshold'] = (
            (df['amount'] > threshold * 0.9) & 
            (df['amount'] < self.aml_config['transaction_threshold'])
        )
        
        # Calculate number of transactions just below threshold in rolling window
        df['structuring_count'] = df.groupby('sender_id')['is_below_threshold'].rolling(
            window, on='transaction_date'
        ).sum().reset_index(0, drop=True)
        
        # Flag potential structuring
        df['potential_structuring'] = df['structuring_count'] >= 3
        
        return df
    
    def detect_suspicious_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect suspicious transactions based on various rules.
        
        Args:
            df (pd.DataFrame): DataFrame with AML features
            
        Returns:
            pd.DataFrame: DataFrame with suspicious transaction flags
        """
        # Initialize suspicious flag
        df['is_suspicious'] = False
        df['suspicious_reasons'] = ''
        
        # Rule 1: Large transactions
        large_tx = df['amount'] > self.aml_config['transaction_threshold']
        df.loc[large_tx, 'suspicious_reasons'] += 'large_transaction;'
        
        # Rule 2: Potential structuring
        structuring = df['potential_structuring'] == True
        df.loc[structuring, 'suspicious_reasons'] += 'potential_structuring;'
        
        # Rule 3: High-risk country involvement
        high_risk = df['sender_high_risk'] | df['recipient_high_risk']
        df.loc[high_risk, 'suspicious_reasons'] += 'high_risk_country;'
        
        # Rule 4: Unusual activity (outside 3 standard deviations)
        unusual_amount = (
            (df['amount'] > df['rolling_avg_amount'] + 3 * df['rolling_std_amount']) |
            (df['daily_tx_count'] > df['daily_tx_count'].mean() + 3 * df['daily_tx_count'].std())
        )
        df.loc[unusual_amount, 'suspicious_reasons'] += 'unusual_activity;'
        
        # Mark as suspicious if any rules were triggered
        df['is_suspicious'] = df['suspicious_reasons'] != ''
        
        return df
    
    def _calculate_country_risk_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk scores for countries based on transaction patterns."""
        # This is a simplified example - in practice, you'd want more sophisticated scoring
        country_stats = df.groupby('sender_country').agg({
            'amount': ['mean', 'std', 'count'],
            'is_suspicious': 'mean'
        })
        
        # Normalize and combine metrics for a risk score
        risk_scores = {}
        for country in country_stats.index:
            avg_amount = country_stats.loc[country, ('amount', 'mean')]
            suspicious_ratio = country_stats.loc[country, ('is_suspicious', 'mean')]
            tx_count = country_stats.loc[country, ('amount', 'count')]
            
            # Simple risk score calculation (customize based on your needs)
            risk_score = (
                0.4 * suspicious_ratio +
                0.3 * (avg_amount / avg_amount.max()) +
                0.3 * (tx_count / tx_count.max())
            )
            
            risk_scores[country] = risk_score
            
        return risk_scores

    def detect_anomalies_isolation_forest(self, df: pd.DataFrame, contamination: float = 0.1) -> pd.DataFrame:
        """
        Detect anomalies using Isolation Forest algorithm.
        
        Args:
            df (pd.DataFrame): DataFrame with transaction features
            contamination (float): Expected proportion of anomalies
            
        Returns:
            pd.DataFrame: DataFrame with anomaly predictions
        """
        # Select numerical features for anomaly detection
        features = [
            'amount', 'daily_tx_count', 'daily_total_amount',
            'daily_avg_amount', 'rolling_avg_amount', 'rolling_std_amount',
            'sender_country_risk_score', 'recipient_country_risk_score'
        ]
        
        # Initialize and train Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        # Fit and predict
        df['is_anomaly'] = iso_forest.fit_predict(df[features])
        # Convert predictions: -1 for anomalies, 1 for normal points
        df['is_anomaly'] = df['is_anomaly'].map({1: 0, -1: 1})
        
        # Calculate anomaly scores
        df['anomaly_score'] = -iso_forest.score_samples(df[features])
        
        logger.info(f"Detected {df['is_anomaly'].sum()} anomalies using Isolation Forest")
        return df
    
    def train_supervised_model(self, df: pd.DataFrame, 
                             test_size: float = 0.2) -> Tuple[RandomForestClassifier, dict]:
        """
        Train a supervised model using Random Forest for suspicious transaction detection.
        
        Args:
            df (pd.DataFrame): DataFrame with labeled data
            test_size (float): Proportion of data to use for testing
            
        Returns:
            Tuple[RandomForestClassifier, dict]: Trained model and performance metrics
        """
        # Prepare features
        features = [
            'amount', 'daily_tx_count', 'daily_total_amount',
            'daily_avg_amount', 'rolling_avg_amount', 'rolling_std_amount',
            'sender_country_risk_score', 'recipient_country_risk_score',
            'structuring_count', 'is_cross_border', 'sender_high_risk',
            'recipient_high_risk'
        ]
        
        X = df[features]
        y = df['is_suspicious']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = rf_model.predict(X_test)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Generate performance metrics
        metrics = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': feature_importance
        }
        
        logger.info("Random Forest model training completed")
        logger.info("\nClassification Report:\n" + metrics['classification_report'])
        
        return rf_model, metrics
    
    def build_transaction_network(self, df: pd.DataFrame, 
                                min_transactions: int = 5,
                                min_amount: float = 10000) -> nx.Graph:
        """
        Build and analyze transaction network using NetworkX.
        
        Args:
            df (pd.DataFrame): Transaction data
            min_transactions (int): Minimum number of transactions to include edge
            min_amount (float): Minimum total amount to include edge
            
        Returns:
            nx.Graph: Transaction network graph
        """
        # Create directed graph
        G = nx.DiGraph()
        
        # Aggregate transactions between parties
        edge_data = df.groupby(['sender_id', 'recipient_id']).agg({
            'amount': ['sum', 'count'],
            'is_suspicious': 'sum'
        }).reset_index()
        
        # Add edges meeting criteria
        for _, row in edge_data.iterrows():
            if row[('amount', 'count')] >= min_transactions and row[('amount', 'sum')] >= min_amount:
                G.add_edge(
                    row['sender_id'],
                    row['recipient_id'],
                    weight=row[('amount', 'sum')],
                    transaction_count=row[('amount', 'count')],
                    suspicious_count=row[('is_suspicious', 'sum')]
                )
        
        # Calculate network metrics
        metrics = self._calculate_network_metrics(G)
        
        # Identify high-risk nodes and communities
        high_risk_nodes = self._identify_high_risk_nodes(G)
        communities = self._detect_communities(G)
        
        logger.info(f"Network analysis complete: {len(G.nodes)} nodes, {len(G.edges)} edges")
        logger.info(f"Identified {len(high_risk_nodes)} high-risk nodes")
        logger.info(f"Detected {len(communities)} distinct communities")
        
        return G, metrics, high_risk_nodes, communities
    
    def _calculate_network_metrics(self, G: nx.Graph) -> dict:
        """Calculate various network metrics for analysis."""
        metrics = {
            'density': nx.density(G),
            'average_clustering': nx.average_clustering(G),
            'degree_centrality': nx.degree_centrality(G),
            'betweenness_centrality': nx.betweenness_centrality(G),
            'pagerank': nx.pagerank(G)
        }
        return metrics
    
    def _identify_high_risk_nodes(self, G: nx.Graph) -> Set[str]:
        """Identify high-risk nodes based on network metrics."""
        high_risk_nodes = set()
        
        # Get centrality metrics
        degree_cent = nx.degree_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G)
        pagerank = nx.pagerank(G)
        
        # Identify nodes with high metrics
        for node in G.nodes():
            risk_score = (
                degree_cent[node] +
                betweenness_cent[node] +
                pagerank[node]
            ) / 3
            
            if risk_score > 0.8:  # Threshold can be adjusted
                high_risk_nodes.add(node)
        
        return high_risk_nodes
    
    def _detect_communities(self, G: nx.Graph) -> List[Set[str]]:
        """Detect communities in the transaction network."""
        # Convert to undirected graph for community detection
        G_undirected = G.to_undirected()
        
        # Use Louvain method for community detection
        communities = nx.community.louvain_communities(G_undirected)
        return communities

def main():
    """Example usage of the ExcelProcessor class with advanced analytics."""
    file_path = "transaction_data.xlsx"
    processor = ExcelProcessor(file_path)
    
    # Process the file in chunks with basic AML features
    all_data = []
    total_rows = 0
    
    logger.info("Starting transaction data processing...")
    
    for chunk in processor.read_in_chunks():
        # Add AML features
        chunk = processor.add_aml_features(chunk)
        chunk = processor.detect_suspicious_transactions(chunk)
        all_data.append(chunk)
        total_rows += len(chunk)
    
    # Combine all chunks
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Perform anomaly detection
    logger.info("Performing anomaly detection...")
    final_df = processor.detect_anomalies_isolation_forest(final_df)
    
    # Train supervised model if labeled data is available
    if 'is_suspicious' in final_df.columns:
        logger.info("Training supervised model...")
        model, metrics = processor.train_supervised_model(final_df)
        
        # Save model performance metrics
        with open('model_metrics.json', 'w') as f:
            json.dump({
                'classification_report': metrics['classification_report'],
                'feature_importance': metrics['feature_importance'].to_dict()
            }, f, indent=4)
    
    # Perform network analysis
    logger.info("Building transaction network...")
    G, network_metrics, high_risk_nodes, communities = processor.build_transaction_network(final_df)
    
    # Save network analysis results
    nx.write_gexf(G, 'transaction_network.gexf')  # Can be visualized with Gephi
    
    # Save high-risk entities to separate files
    suspicious_df = final_df[
        (final_df['is_suspicious']) | 
        (final_df['is_anomaly']) | 
        (final_df['sender_id'].isin(high_risk_nodes))
    ].copy()
    
    suspicious_df.to_excel('suspicious_transactions.xlsx', index=False)
    
    # Generate summary report
    summary = {
        'total_transactions': len(final_df),
        'suspicious_transactions': suspicious_df['is_suspicious'].sum(),
        'anomalies_detected': suspicious_df['is_anomaly'].sum(),
        'high_risk_nodes': len(high_risk_nodes),
        'network_communities': len(communities),
        'network_density': network_metrics['density']
    }
    
    with open('analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info("Analysis complete. Results saved to files.")
    return final_df, G, summary

if __name__ == "__main__":
    main() 