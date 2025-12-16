# ml/features.py - Feature extraction for NN

import networkx as nx
import numpy as np

class PathFeatureExtractor:
    """Extract features from candidate paths for ML routing"""
    
    def __init__(self, G):
        self.G = G
    
    def extract_features(self, flow, path):
        """
        Extract features for a given path
        Returns: numpy array of features
        """
        if not path or len(path) < 2:
            return None
        
        features = []
        
        # 1. Path length (normalized by network diameter)
        diameter = nx.diameter(self.G)
        path_length = len(path) - 1
        features.append(path_length / diameter)
        
        # 2. Total delay estimate
        total_delay = 0
        min_bandwidth = float('inf')
        total_jitter = 0
        avg_loss = 0
        queue_depth = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge = self.G[u][v]
            
            total_delay += edge.get('current_delay', edge['delay'])
            min_bandwidth = min(min_bandwidth, edge.get('available_bw', edge['bandwidth']))
            total_jitter += edge['jitter']
            avg_loss += edge['loss']
            
            # Queue depth if available
            if 'link_obj' in edge:
                queue_depth += len(edge['link_obj'].current_flows)
        
        avg_loss /= path_length
        
        features.append(total_delay / 100.0)  # normalize (max ~100ms)
        features.append(min_bandwidth / 100.0)  # normalize (max 100 Mbps)
        features.append(total_jitter / 50.0)  # normalize
        features.append(avg_loss / 5.0)  # normalize (max 5%)
        features.append(queue_depth / 10.0)  # normalize
        
        # 3. Flow-specific features
        features.append(flow.deadline_ms / 150.0)  # normalize by max deadline
        features.append(flow.bandwidth_mbps / 10.0)  # normalize
        features.append(flow.priority / 3.0)  # normalize (1-3)
        
        # 4. Time urgency (how much time left until deadline)
        time_to_deadline = flow.deadline_ms - total_delay
        features.append(max(0, time_to_deadline) / flow.deadline_ms)
        
        # 5. Bandwidth feasibility
        bw_ratio = min_bandwidth / flow.bandwidth_mbps if flow.bandwidth_mbps > 0 else 1.0
        features.append(min(1.0, bw_ratio))
        
        # 6. Load balancing score (how evenly utilized is this path?)
        link_utilizations = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge = self.G[u][v]
            util = 1 - (edge.get('available_bw', edge['bandwidth']) / edge['bandwidth'])
            link_utilizations.append(util)

        # Variance in utilization (lower = better load balance)
        util_variance = np.var(link_utilizations) if link_utilizations else 0
        features.append(util_variance)

        # 7. Hop-by-hop delay distribution (is delay evenly spread?)
        hop_delays = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            hop_delays.append(self.G[u][v].get('current_delay', self.G[u][v]['delay']))

        delay_variance = np.var(hop_delays) if hop_delays else 0
        features.append(delay_variance / 100.0)

        
        return np.array(features, dtype=np.float32)
    
    def get_candidate_paths(self, source, destination, k=3):
        """Get k candidate paths between source and destination"""
        try:
            # Get k shortest paths
            paths = list(nx.shortest_simple_paths(self.G, source, destination))
            return paths[:k]
        except nx.NetworkXNoPath:
            return []