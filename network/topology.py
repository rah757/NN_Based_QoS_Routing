# network/topology.py

import networkx as nx
import random
import math

class WANTopology:
    """Creates and manages WAN topology with realistic link properties"""
    
    def __init__(self, n_nodes=20, seed=42):
        """
        Args:
            n_nodes: Number of routers in the WAN
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        self.n_nodes = n_nodes
        self.G = nx.Graph()
        
        self._create_topology()
        self._assign_link_properties()
    
    def _create_topology(self):
        """Create WAN topology using Waxman model (geographic-based)"""
        # Add nodes
        for i in range(self.n_nodes):
            # Assign random geographic coordinates
            x = random.uniform(0, 1000)  # km
            y = random.uniform(0, 1000)  # km
            self.G.add_node(i, pos=(x, y))
        
        # Create edges using Waxman model
        # P(edge) = β * exp(-d / (α * L))
        # where d = distance, L = max distance, α,β = parameters
        
        positions = nx.get_node_attributes(self.G, 'pos')
        max_dist = math.sqrt(2) * 1000  # diagonal of 1000x1000 grid
        
        alpha = 0.4  # controls distance effect
        beta = 0.3   # controls density
        
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                # Calculate distance
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Waxman probability
                prob = beta * math.exp(-distance / (alpha * max_dist))
                
                if random.random() < prob:
                    self.G.add_edge(i, j, distance=distance)
        
        # Ensure connectivity - if disconnected, add edges
        if not nx.is_connected(self.G):
            components = list(nx.connected_components(self.G))
            # Connect components
            for i in range(len(components) - 1):
                node1 = random.choice(list(components[i]))
                node2 = random.choice(list(components[i + 1]))
                
                x1, y1 = positions[node1]
                x2, y2 = positions[node2]
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                self.G.add_edge(node1, node2, distance=distance)
    
    def _assign_link_properties(self):
        """Assign realistic bandwidth, delay, jitter, loss to each link"""
        for u, v in self.G.edges():
            distance = self.G[u][v]['distance']
            
            # Bandwidth: 10-100 Mbps (inversely related to distance)
            # Longer links tend to have lower bandwidth
            bw_factor = 1.0 - (distance / 1500)  # normalize
            bw_factor = max(0.1, bw_factor)  # at least 0.1
            bandwidth = random.uniform(10, 100) * bw_factor
            
            # Propagation delay: based on distance (light speed ~200km/ms in fiber)
            prop_delay = distance / 200.0  # ms
            
            # Queuing delay: 1-10ms depending on expected load
            queue_delay = random.uniform(1, 10)
            
            # Total delay
            delay = prop_delay + queue_delay
            
            # Jitter: 1-10ms (higher for longer/slower links)
            jitter = random.uniform(1, 10) * (1 + distance/1000)
            
            # Packet loss: 0-5% (higher for longer/congested links)
            loss = random.uniform(0, 5) * (1 + distance/1000) / 2
            loss = min(loss, 5.0)  # cap at 5%
            
            # Assign to edge
            self.G[u][v].update({
                'bandwidth': bandwidth,      # Mbps
                'delay': delay,              # ms
                'jitter': jitter,            # ms
                'loss': loss,                # percentage
                'available_bw': bandwidth,   # current available (starts at max)
                'queue_length': 0            # current packets in queue
            })
    
    def get_graph(self):
        """Return the NetworkX graph"""
        return self.G
    
    def get_link_info(self, u, v):
        """Get properties of a specific link"""
        if not self.G.has_edge(u, v):
            return None
        return self.G[u][v]
    
    def print_statistics(self):
        """Print topology statistics"""
        print(f"\nTopology Statistics:")
        print(f"  Nodes: {self.G.number_of_nodes()}")
        print(f"  Edges: {self.G.number_of_edges()}")
        print(f"  Average degree: {sum(dict(self.G.degree()).values()) / self.G.number_of_nodes():.2f}")
        print(f"  Connected: {nx.is_connected(self.G)}")
        print(f"  Diameter: {nx.diameter(self.G) if nx.is_connected(self.G) else 'N/A'}")
        
        # Link statistics
        bandwidths = [d['bandwidth'] for u, v, d in self.G.edges(data=True)]
        delays = [d['delay'] for u, v, d in self.G.edges(data=True)]
        losses = [d['loss'] for u, v, d in self.G.edges(data=True)]
        
        print(f"\n  Bandwidth: {min(bandwidths):.1f} - {max(bandwidths):.1f} Mbps (avg: {sum(bandwidths)/len(bandwidths):.1f})")
        print(f"  Delay: {min(delays):.1f} - {max(delays):.1f} ms (avg: {sum(delays)/len(delays):.1f})")
        print(f"  Loss: {min(losses):.2f} - {max(losses):.2f} % (avg: {sum(losses)/len(losses):.2f})")