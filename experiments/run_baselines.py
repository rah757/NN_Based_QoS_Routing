# experiments/run_baselines.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import simpy
from network.topology import WANTopology
from traffic.generator import TrafficGenerator
from routing.dijkstra import DijkstraRouter
from routing.widest_path import WidestPathRouter
from routing.min_delay import MinDelayRouter
from simulation.simulator import NetworkSimulator
from simulation.metrics import MetricsCollector

def run_experiment(algorithm_name, router_class, load_factor=0.6, duration=30):
    """Run a single experiment"""
    print(f"\nRunning {algorithm_name} (load={load_factor*100:.0f}%)...")
    
    env = simpy.Environment()
    topology = WANTopology(n_nodes=20, seed=42)
    router = router_class()
    traffic_gen = TrafficGenerator(env, topology.get_graph(), 
                                   arrival_rate=5, load_factor=load_factor)
    
    sim = NetworkSimulator(env, topology, traffic_gen, router)
    sim.run()
    env.run(until=duration)
    
    return sim.get_metrics()

def run_all_baselines():
    """Run all baseline algorithms"""
    
    print("="*80)
    print("BASELINE ALGORITHM COMPARISON")
    print("="*80)
    
    algorithms = {
        'Dijkstra': DijkstraRouter,
        'Widest-Path': WidestPathRouter,
        'Min-Delay': MinDelayRouter
    }
    
    collector = MetricsCollector()
    
    # Test different load levels
    for load in [0.3, 0.6, 0.9]:
        print(f"\n{'='*80}")
        print(f"LOAD FACTOR: {load*100:.0f}%")
        print(f"{'='*80}")
        
        for algo_name, router_class in algorithms.items():
            metrics = run_experiment(algo_name, router_class, 
                                    load_factor=load, duration=30)
            collector.add_run(f"{algo_name} ({load*100:.0f}%)", metrics)
    
    # Show comparison
    collector.compare_algorithms()
    collector.plot_comparison(save_path='baseline_comparison.png')
    
    return collector

if __name__ == "__main__":
    collector = run_all_baselines()