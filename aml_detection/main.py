"""Main entry point for AML detection system."""
import pandas as pd
import json
import networkx as nx
import argparse
import sys
from pathlib import Path
from processors.excel_processor import ExcelProcessor
from processors.feature_processor import FeatureProcessor
from processors.network_processor import NetworkProcessor
from models.anomaly_detector import AnomalyDetector
from models.supervised_model import SuspiciousTransactionClassifier
from utils.logging_utils import logger
from typing import Tuple

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AML Detection System - Analyze transaction data for suspicious patterns'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the Excel file containing transaction data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='aml_results',
        help='Directory to save analysis results (default: aml_results)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100000,
        help='Number of rows to process at once (default: 100000)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    if not input_path.suffix.lower() in ['.xlsx', '.xls']:
        logger.error(f"Input file must be an Excel file (.xlsx or .xls)")
        sys.exit(1)
    
    return args

def save_model_metrics(metrics: dict, output_dir: Path):
    """Save model performance metrics."""
    metrics_file = output_dir / 'model_metrics.json'
    
    # Convert non-serializable objects to serializable format
    serializable_metrics = {
        'classification_report': metrics['classification_report'],
        'confusion_matrix': metrics['confusion_matrix'],
        'feature_importance': metrics['feature_importance'].to_dict(),
        'accuracy': metrics['accuracy']
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    
    logger.info(f"Model metrics saved to {metrics_file}")

def save_results(df: pd.DataFrame, G: nx.DiGraph, network_metrics: dict, 
                high_risk_nodes: set, communities: list, output_dir: Path):
    """Save analysis results to files."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save suspicious transactions
    suspicious_df = df[
        (df['is_suspicious']) | 
        (df['is_anomaly']) | 
        (df['sender_id'].isin(high_risk_nodes))
    ].copy()
    
    suspicious_file = output_dir / 'suspicious_transactions.xlsx'
    suspicious_df.to_excel(suspicious_file, index=False)
    
    # Save network graph
    network_file = output_dir / 'transaction_network.gexf'
    nx.write_gexf(G, network_file)
    
    # Generate and save summary report
    summary = {
        'total_transactions': len(df),
        'suspicious_transactions': suspicious_df['is_suspicious'].sum(),
        'anomalies_detected': suspicious_df['is_anomaly'].sum(),
        'high_risk_nodes': len(high_risk_nodes),
        'network_communities': len(communities),
        'network_metrics': {
            k: float(v) if isinstance(v, (int, float)) else str(v)
            for k, v in network_metrics.items()
            if k in ['density', 'average_clustering', 'avg_degree', 'max_degree']
        }
    }
    
    summary_file = output_dir / 'analysis_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"Results saved to {output_dir}")

def run_analysis(input_file: Path, output_dir: Path, chunk_size: int = 100000) -> Tuple[pd.DataFrame, nx.DiGraph]:
    """
    Run the AML detection analysis pipeline.
    
    Args:
        input_file (Path): Path to input Excel file
        output_dir (Path): Path to output directory
        chunk_size (int): Number of rows to process at once
    
    Returns:
        Tuple[pd.DataFrame, nx.DiGraph]: Processed dataframe and network graph
    """
    # Initialize processors
    excel_proc = ExcelProcessor(str(input_file))
    feature_proc = FeatureProcessor()
    network_proc = NetworkProcessor()
    anomaly_detector = AnomalyDetector()
    classifier = SuspiciousTransactionClassifier()
    
    try:
        # Process data
        logger.info(f"Starting data processing for {input_file}")
        all_data = []
        total_rows = 0
        
        for chunk in excel_proc.read_in_chunks(chunk_size=chunk_size):
            processed_chunk = feature_proc.process_features(chunk)
            all_data.append(processed_chunk)
            total_rows += len(chunk)
            logger.info(f"Processed {total_rows} rows so far...")
        
        # Combine all chunks
        final_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Completed processing {len(final_df)} transactions")
        
        # Perform analysis
        logger.info("Running anomaly detection...")
        final_df = anomaly_detector.detect(final_df)
        
        if 'is_suspicious' in final_df.columns:
            logger.info("Training supervised model...")
            model, metrics = classifier.train(final_df)
            save_model_metrics(metrics, output_dir)
        
        logger.info("Performing network analysis...")
        G, network_metrics, high_risk_nodes, communities = network_proc.build_network(final_df)
        
        # Save results
        save_results(final_df, G, network_metrics, high_risk_nodes, communities, output_dir)
        
        logger.info("Analysis pipeline completed successfully")
        return final_df, G
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {str(e)}")
        raise

def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    input_file = Path(args.input_file)
    try:
        final_df, G = run_analysis(
            input_file=input_file,
            output_dir=output_dir,
            chunk_size=args.chunk_size
        )
        
        logger.info(f"""
        Analysis Summary:
        - Input file: {input_file}
        - Total transactions: {len(final_df)}
        - Suspicious transactions: {final_df['is_suspicious'].sum()}
        - Anomalies detected: {final_df['is_anomaly'].sum()}
        - Results saved to: {output_dir}
        """)
        
        return final_df, G
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 