# experiments/run_ml_comparison.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import simpy
from network.topology import WANTopology
from traffic.generator import TrafficGenerator
from routing.dijkstra import DijkstraRouter
from routing.widest_path import WidestPathRouter
from routing.min_delay import MinDelayRouter
from routing.nn_router import NNRouter
from simulation.simulator import NetworkSimulator
from simulation.metrics import MetricsCollector

def run_experiment(algorithm_name, router, load_factor=0.6, duration=30):
    """Run a single experiment"""
    print(f"\nRunning {algorithm_name} (load={load_factor*100:.0f}%)...")
    
    env = simpy.Environment()
    topology = WANTopology(n_nodes=20, seed=42)
    traffic_gen = TrafficGenerator(env, topology.get_graph(), 
                                   arrival_rate=5, load_factor=load_factor)
    
    sim = NetworkSimulator(env, topology, traffic_gen, router)
    sim.run()
    env.run(until=duration)
    
    return sim.get_metrics()

def main():
    """Compare NN router against baselines"""
    
    print("="*80)
    print("ML ROUTER vs BASELINE COMPARISON")
    print("="*80)
    
    # Create routers
    routers = {
        'Dijkstra': DijkstraRouter(),
        'Widest-Path': WidestPathRouter(),
        'Min-Delay': MinDelayRouter(),
        'NN-Router': NNRouter(pretrained_model='models/nn_router.pth', epsilon=0.1)
    }
    
    collector = MetricsCollector()
    
    # Test at different loads
    for load in [0.3, 0.6, 0.9]:
        print(f"\n{'='*80}")
        print(f"LOAD FACTOR: {load*100:.0f}%")
        print(f"{'='*80}")
        
        for algo_name, router in routers.items():
            metrics = run_experiment(algo_name, router, 
                                    load_factor=load, duration=30)
            collector.add_run(f"{algo_name} ({load*100:.0f}%)", metrics)
            
            # Print quick summary
            print(f"  Miss ratio: {metrics['deadline_miss_ratio']*100:.1f}%, "
                  f"Avg delay: {metrics['avg_delay']:.1f}ms, "
                  f"Dropped: {metrics['dropped_flows']}")
    
    # Show comparison
    print("\n")
    collector.compare_algorithms()
    collector.plot_comparison(save_path='ml_vs_baseline.png')
    
    # Detailed breakdown by traffic class
    print("\n" + "="*80)
    print("BREAKDOWN BY TRAFFIC CLASS")
    print("="*80)
    
    for load in [0.6]:  # Focus on medium load
        print(f"\nLoad: {load*100:.0f}%")
        for algo_name, router in routers.items():
            metrics = run_experiment(algo_name, router, load_factor=load, duration=30)
            
            print(f"\n{algo_name}:")
            if metrics['by_class']:
                for tc, stats in metrics['by_class'].items():
                    print(f"  {tc:12s}: miss={stats['miss_ratio']*100:5.1f}%, "
                          f"delay={stats['avg_delay']:5.1f}ms, count={stats['count']}")

if __name__ == "__main__":
    main()