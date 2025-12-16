# routing/dijkstra.py - Simple baseline router

import networkx as nx

class DijkstraRouter:
    """Simple shortest-path router (hop count minimization)"""
    
    def find_path(self, flow, G):
        """Find shortest path for a flow"""
        try:
            path = nx.shortest_path(G, flow.source, flow.destination)
            return path
        except nx.NetworkXNoPath:
            return None