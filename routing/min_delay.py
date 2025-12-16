# routing/min_delay.py

import networkx as nx

class MinDelayRouter:
    """Minimum delay routing: minimize cumulative delay"""
    
    def find_path(self, flow, G):
        """Find path with minimum total delay"""
        try:
            # Use delay as weight
            def weight_fn(u, v, d):
                return d['delay']
            
            path = nx.shortest_path(G, flow.source, flow.destination,
                                   weight=weight_fn)
            return path
        except nx.NetworkXNoPath:
            return None