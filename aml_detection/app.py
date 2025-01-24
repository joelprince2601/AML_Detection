"""Streamlit interface for AML Detection System."""
import streamlit as st
import pandas as pd
import json
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path
import sys
import io
from datetime import datetime
import traceback

# Add parent directory to path to fix imports
sys.path.append(str(Path(__file__).parent.parent))

from aml_detection.processors.excel_processor import ExcelProcessor
from aml_detection.processors.feature_processor import FeatureProcessor
from aml_detection.processors.network_processor import NetworkProcessor
from aml_detection.models.anomaly_detector import AnomalyDetector
from aml_detection.models.supervised_model import SuspiciousTransactionClassifier
from aml_detection.utils.logging_utils import logger

def create_network_graph(G, high_risk_nodes):
    """Create interactive network visualization."""
    pos = nx.spring_layout(G)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if node in high_risk_nodes:
            node_colors.append('red')
            node_text.append(f"High Risk Node: {node}")
        else:
            node_colors.append('blue')
            node_text.append(f"Node: {node}")

    # Create edges trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create nodes trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            size=10,
            color=node_colors,
            line_width=2))
    
    node_trace.text = node_text

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0,l=0,r=0,t=0),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig

def save_uploadedfile(uploadedfile):
    """Save uploaded file temporarily and return path."""
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / f"temp_{uploadedfile.name}"
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return str(file_path)

@st.cache_data
def load_example_data():
    """Load example dataset if available."""
    example_path = Path(__file__).parent / "example_data" / "sample_transactions.xlsx"
    if example_path.exists():
        return pd.read_excel(example_path)
    return None

def run_analysis(input_data, progress_bar) -> tuple:
    """
    Run AML analysis pipeline with progress tracking.
    
    Args:
        input_data: Either a file path (str) or pandas DataFrame
        progress_bar: Streamlit progress bar
    
    Returns:
        tuple: (final_df, G, metrics, network_metrics, high_risk_nodes, communities)
    """
    try:
        # Initialize processors
        feature_proc = FeatureProcessor()
        network_proc = NetworkProcessor()
        anomaly_detector = AnomalyDetector()
        classifier = SuspiciousTransactionClassifier()
        
        # Process data
        all_data = []
        processed_rows = 0
        
        if isinstance(input_data, (str, Path)):
            # Process Excel file in chunks
            excel_proc = ExcelProcessor(str(input_data))
            total_rows = excel_proc._count_rows()
            
            for chunk in excel_proc.read_in_chunks():
                processed_chunk = feature_proc.process_features(chunk)
                all_data.append(processed_chunk)
                processed_rows += len(chunk)
                
                # Update progress bar
                progress = min(processed_rows / total_rows, 1.0)
                progress_bar.progress(progress)
            
            # Combine all chunks
            final_df = pd.concat(all_data, ignore_index=True)
            
        else:
            # Input is already a DataFrame
            final_df = input_data.copy()
            progress_bar.progress(0.5)  # Show some progress
            final_df = feature_proc.process_features(final_df)
            progress_bar.progress(1.0)
        
        # Perform analysis
        logger.info("Running anomaly detection...")
        final_df = anomaly_detector.detect(final_df)
        
        if 'is_suspicious' in final_df.columns:
            logger.info("Training supervised model...")
            model, metrics = classifier.train(final_df)
        else:
            metrics = None
        
        logger.info("Performing network analysis...")
        G, network_metrics, high_risk_nodes, communities = network_proc.build_network(final_df)
        
        return final_df, G, metrics, network_metrics, high_risk_nodes, communities
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise

def main():
    """Streamlit app main function."""
    st.set_page_config(page_title="AML Detection System", layout="wide")
    
    st.title("AML Transaction Analysis System")
    
    # Sidebar
    st.sidebar.header("Settings")
    use_example = st.sidebar.checkbox("Use Example Dataset", False)
    
    # Main content
    st.write("""
    ## Transaction Data Analysis
    Upload your transaction data Excel file for analysis or use the example dataset. 
    
    ### Required Columns:
    - transaction_date
    - amount
    - sender_id
    - sender_country
    - recipient_id
    - recipient_country
    """)
    
    input_data = None
    
    # File upload or example data
    if use_example:
        input_data = load_example_data()
        if input_data is None:
            st.error("Example dataset not found!")
            return
    else:
        uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
        if uploaded_file is None:
            return
        
        try:
            input_data = save_uploadedfile(uploaded_file)
        except Exception as e:
            st.error(f"Error saving uploaded file: {str(e)}")
            return
    
    if input_data is not None:
        try:
            with st.spinner('Analyzing transactions...'):
                progress_bar = st.progress(0)
                
                # Run analysis
                final_df, G, metrics, network_metrics, high_risk_nodes, communities = run_analysis(
                    input_data,
                    progress_bar
                )
                
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Summary", "Suspicious Transactions", 
                    "Network Analysis", "Model Performance"
                ])
                
                with tab1:
                    st.header("Analysis Summary")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Transactions", len(final_df))
                        st.metric("Suspicious Transactions", int(final_df['is_suspicious'].sum()))
                        st.metric("Anomalies Detected", int(final_df['is_anomaly'].sum()))
                    
                    with col2:
                        st.metric("High Risk Nodes", len(high_risk_nodes))
                        st.metric("Network Communities", len(communities))
                        st.metric("Network Density", round(float(network_metrics['density']), 4))
                
                with tab2:
                    st.header("Suspicious Transactions")
                    suspicious_df = final_df[
                        (final_df['is_suspicious']) | 
                        (final_df['is_anomaly']) | 
                        (final_df['sender_id'].isin(high_risk_nodes))
                    ].copy()
                    
                    st.dataframe(suspicious_df)
                    
                    # Download button
                    output = io.BytesIO()
                    suspicious_df.to_excel(output, index=False)
                    st.download_button(
                        label="Download Suspicious Transactions",
                        data=output.getvalue(),
                        file_name="suspicious_transactions.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                
                with tab3:
                    st.header("Network Analysis")
                    
                    # Display network visualization
                    fig = create_network_graph(G, high_risk_nodes)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display network metrics
                    st.subheader("Network Metrics")
                    st.json({k: float(v) if isinstance(v, (int, float)) else v 
                            for k, v in network_metrics.items()
                            if k in ['density', 'average_clustering', 'avg_degree', 'max_degree']})
                
                with tab4:
                    st.header("Model Performance")
                    if metrics:
                        st.subheader("Classification Report")
                        st.text(metrics['classification_report'])
                        
                        st.subheader("Feature Importance")
                        st.dataframe(metrics['feature_importance'])
                    else:
                        st.info("No supervised model metrics available (requires labeled data)")
            
            st.success("Analysis completed successfully!")
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.error(traceback.format_exc())
        
        finally:
            # Cleanup
            if not use_example and 'input_data' in locals():
                Path(input_data).unlink(missing_ok=True)
                Path("temp").rmdir()

if __name__ == "__main__":
    main() 