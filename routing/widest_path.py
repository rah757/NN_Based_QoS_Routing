# routing/widest_path.py

import networkx as nx

class WidestPathRouter:
    """Widest-shortest path: maximize minimum bandwidth"""
    
    def find_path(self, flow, G):
        """Find path with maximum minimum bandwidth"""
        try:
            # Create weight based on inverse bandwidth
            def weight_fn(u, v, d):
                return 1.0 / (d['available_bw'] + 0.1)  # avoid division by zero
            
            path = nx.shortest_path(G, flow.source, flow.destination, 
                                   weight=weight_fn)
            return path
        except nx.NetworkXNoPath:
            return None