# test_topology.py - Test the WAN topology

import matplotlib.pyplot as plt
import networkx as nx
from network.topology import WANTopology

def test_topology():
    """Test WAN topology creation and visualization"""
    
    print("Creating WAN topology...")
    wan = WANTopology(n_nodes=20, seed=42)
    G = wan.get_graph()
    
    wan.print_statistics()
    
    # Test shortest path
    print(f"\nShortest path (0 → 10): {nx.shortest_path(G, 0, 10)}")
    
    # Test link properties
    print("\nSample link properties:")
    for i, (u, v) in enumerate(list(G.edges())[:5]):
        info = wan.get_link_info(u, v)
        print(f"  Link {u}↔{v}: BW={info['bandwidth']:.1f}Mbps, Delay={info['delay']:.1f}ms, Loss={info['loss']:.2f}%")
    
    # Visualize
    visualize_topology(G)
    
    return wan

def visualize_topology(G):
    """Visualize the WAN topology"""
    plt.figure(figsize=(12, 8))
    
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.9)
    
    # Draw edges with color based on bandwidth
    edges = G.edges()
    bandwidths = [G[u][v]['bandwidth'] for u, v in edges]
    
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6,
                          edge_color=bandwidths, edge_cmap=plt.cm.Blues)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title("WAN Topology (Edge color = bandwidth)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('topology.png', dpi=150)
    print("\nTopology visualization saved to 'topology.png'")
    plt.close()

if __name__ == "__main__":
    test_topology()