"""Network analysis module."""
import networkx as nx
from typing import Tuple, Set, List, Dict
import pandas as pd
import numpy as np
from ..utils.logging_utils import logger

class NetworkProcessor:
    def __init__(self, min_transactions: int = 5, min_amount: float = 10000):
        self.min_transactions = min_transactions
        self.min_amount = min_amount
    
    def build_network(self, df: pd.DataFrame) -> Tuple[nx.DiGraph, dict, Set[str], List[Set[str]]]:
        """Build and analyze transaction network."""
        required_cols = {'sender_id', 'recipient_id', 'amount'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")
        
        G = self._create_graph(df)
        
        # Handle empty network case
        if len(G.nodes()) == 0:
            logger.warning("No nodes meet the minimum criteria for network analysis")
            return G, self._get_empty_metrics(), set(), []
        
        metrics = self._calculate_network_metrics(G)
        high_risk_nodes = self._identify_high_risk_nodes(G)
        communities = self._detect_communities(G)
        
        logger.info(f"Network analysis complete: {len(G.nodes)} nodes, {len(G.edges)} edges")
        logger.info(f"Identified {len(high_risk_nodes)} high-risk nodes")
        logger.info(f"Detected {len(communities)} distinct communities")
        
        return G, metrics, high_risk_nodes, communities
    
    def _get_empty_metrics(self) -> dict:
        """Return default metrics for empty network."""
        return {
            'density': 0.0,
            'average_clustering': 0.0,
            'degree_centrality': {},
            'betweenness_centrality': {},
            'pagerank': {},
            'avg_degree': 0.0,
            'max_degree': 0.0
        }
    
    def _create_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """Create directed graph from transaction data."""
        G = nx.DiGraph()
        
        try:
            # Aggregate transactions between parties
            edge_data = df.groupby(['sender_id', 'recipient_id']).agg({
                'amount': ['sum', 'count']
            }).reset_index()
            
            # Add edges meeting criteria
            for _, row in edge_data.iterrows():
                if (row[('amount', 'count')] >= self.min_transactions and 
                    row[('amount', 'sum')] >= self.min_amount):
                    G.add_edge(
                        row['sender_id'],
                        row['recipient_id'],
                        weight=row[('amount', 'sum')],
                        transaction_count=row[('amount', 'count')]
                    )
            
            if len(G.nodes()) == 0:
                logger.warning("No transactions meet the minimum criteria for network analysis")
            
            return G
            
        except Exception as e:
            logger.error(f"Error creating network graph: {str(e)}")
            return G
    
    def _calculate_network_metrics(self, G: nx.DiGraph) -> dict:
        """Calculate various network metrics for analysis."""
        try:
            metrics = {}
            
            # Basic metrics that don't require connected graph
            metrics['density'] = nx.density(G)
            
            # Convert to undirected for clustering coefficient
            G_undirected = G.to_undirected()
            
            # Calculate clustering only if there are enough nodes
            if len(G_undirected.nodes()) > 2:
                metrics['average_clustering'] = nx.average_clustering(G_undirected)
            else:
                metrics['average_clustering'] = 0.0
            
            # Centrality measures
            if len(G.nodes()) > 0:
                metrics['degree_centrality'] = nx.degree_centrality(G)
                metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
                metrics['pagerank'] = nx.pagerank(G)
                
                # Degree statistics
                degrees = [d for n, d in G.degree()]
                metrics['avg_degree'] = np.mean(degrees) if degrees else 0.0
                metrics['max_degree'] = max(degrees) if degrees else 0.0
            else:
                metrics.update(self._get_empty_metrics())
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating network metrics: {str(e)}")
            return self._get_empty_metrics()
    
    def _identify_high_risk_nodes(self, G: nx.DiGraph) -> Set[str]:
        """Identify high-risk nodes based on network metrics."""
        try:
            if len(G.nodes()) < 3:  # Need at least 3 nodes for meaningful analysis
                return set()
            
            high_risk_nodes = set()
            
            # Calculate centrality metrics
            degree_cent = nx.degree_centrality(G)
            betweenness_cent = nx.betweenness_centrality(G)
            pagerank = nx.pagerank(G)
            
            # Calculate thresholds (e.g., top 10% for each metric)
            metrics = pd.DataFrame({
                'node': list(G.nodes()),
                'degree': list(degree_cent.values()),
                'betweenness': list(betweenness_cent.values()),
                'pagerank': list(pagerank.values())
            })
            
            # Only calculate thresholds if we have enough nodes
            if len(metrics) >= 10:
                thresholds = {
                    'degree': metrics['degree'].quantile(0.9),
                    'betweenness': metrics['betweenness'].quantile(0.9),
                    'pagerank': metrics['pagerank'].quantile(0.9)
                }
                
                # Identify high-risk nodes
                for node in G.nodes():
                    risk_score = (
                        (degree_cent[node] >= thresholds['degree']) +
                        (betweenness_cent[node] >= thresholds['betweenness']) +
                        (pagerank[node] >= thresholds['pagerank'])
                    )
                    
                    if risk_score >= 2:  # Node is high risk if it exceeds threshold in at least 2 metrics
                        high_risk_nodes.add(node)
            else:
                # For small networks, use simpler criteria
                for node in G.nodes():
                    if degree_cent[node] > 0.5:  # Node connected to more than half of other nodes
                        high_risk_nodes.add(node)
            
            return high_risk_nodes
            
        except Exception as e:
            logger.error(f"Error identifying high-risk nodes: {str(e)}")
            return set()
    
    def _detect_communities(self, G: nx.DiGraph) -> List[Set[str]]:
        """Detect communities in the transaction network."""
        try:
            if len(G.nodes()) < 3:  # Need at least 3 nodes for community detection
                return [set(G.nodes())] if G.nodes() else []
            
            # Convert to undirected graph for community detection
            G_undirected = G.to_undirected()
            
            try:
                communities = nx.community.louvain_communities(G_undirected)
            except (nx.NetworkXError, Exception) as e:
                logger.warning(f"Louvain community detection failed: {str(e)}, falling back to connected components")
                communities = list(nx.connected_components(G_undirected))
            
            return communities
            
        except Exception as e:
            logger.error(f"Error detecting communities: {str(e)}")
            return [] 