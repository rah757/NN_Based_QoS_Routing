# test_traffic.py - Standalone test

import simpy
import networkx as nx
from traffic.generator import TrafficGenerator
from traffic.flow import TrafficClass

def test_traffic_generation():
    """Test traffic generator without full network"""
    
    # Create dummy network (we'll build real one next)
    G = nx.Graph()
    G.add_nodes_from(range(5))  # 5 dummy nodes
    
    # Create SimPy environment
    env = simpy.Environment()
    
    # Create traffic generator (light load)
    traffic_gen = TrafficGenerator(env, G, arrival_rate=5, load_factor=0.3)
    
    # Start generation process
    env.process(traffic_gen.generate_traffic())
    
    # Run for 10 simulated seconds
    env.run(until=10)
    
    # Print statistics
    print("\n" + "="*60)
    print("TRAFFIC GENERATION TEST RESULTS")
    print("="*60)
    
    stats = traffic_gen.get_statistics()
    print(f"\nTotal flows generated: {stats['total_flows']}")
    print(f"Generation rate: {stats['generation_rate']:.2f} flows/sec")
    print(f"Average flow size: {stats['avg_size']:.0f} bytes")
    
    print("\nFlows by traffic class:")
    for tc, count in stats['by_class'].items():
        pct = (count / stats['total_flows']) * 100
        print(f"  {tc.value:12s}: {count:3d} ({pct:5.1f}%)")
    
    print("\nSample flows:")
    for flow in traffic_gen.generated_flows[:5]:
        print(f"  {flow}")
        print(f"    Deadline: {flow.deadline_ms}ms, BW: {flow.bandwidth_mbps:.3f}Mbps, Priority: {flow.priority}")
    
    # Check deadline distribution
    deadlines = [f.deadline_ms for f in traffic_gen.generated_flows]
    print(f"\nDeadline range: {min(deadlines)}ms - {max(deadlines)}ms")
    
    return traffic_gen

if __name__ == "__main__":
    test_traffic_generation()