# test_integration.py - Test everything together

import simpy
from network.topology import WANTopology
from traffic.generator import TrafficGenerator
from routing.dijkstra import DijkstraRouter
from simulation.simulator import NetworkSimulator

def test_integration():
    """Test complete simulation pipeline"""
    
    print("="*60)
    print("INTEGRATED SIMULATION TEST")
    print("="*60)
    
    # Create components
    env = simpy.Environment()
    topology = WANTopology(n_nodes=20, seed=42)
    router = DijkstraRouter()
    traffic_gen = TrafficGenerator(env, topology.get_graph(), 
                                   arrival_rate=5, load_factor=0.3)
    
    # Create simulator
    sim = NetworkSimulator(env, topology, traffic_gen, router)
    sim.run()
    
    # Run simulation
    print("\nRunning simulation for 10 seconds...\n")
    env.run(until=10)
    
    # Print results
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    
    metrics = sim.get_metrics()
    
    print(f"\nOverall:")
    print(f"  Total flows: {metrics['total_flows']}")
    print(f"  Completed: {metrics['completed_flows']}")
    print(f"  Dropped: {metrics['dropped_flows']}")
    print(f"  Deadline miss ratio: {metrics['deadline_miss_ratio']*100:.1f}%")
    print(f"  Average delay: {metrics['avg_delay']:.2f}ms")
    print(f"  Delay range: {metrics['min_delay']:.2f} - {metrics['max_delay']:.2f}ms")
    
    print(f"\nBy traffic class:")
    for tc, stats in metrics['by_class'].items():
        miss_pct = stats['miss_ratio'] * 100
        print(f"  {tc:12s}: {stats['count']:3d} flows, "
              f"miss ratio={miss_pct:5.1f}%, avg delay={stats['avg_delay']:.1f}ms")
    
    return sim

if __name__ == "__main__":
    test_integration()