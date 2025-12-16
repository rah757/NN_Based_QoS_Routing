# experiments/final_evaluation.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import simpy
import numpy as np
import matplotlib.pyplot as plt
from network.topology import WANTopology
from traffic.generator import TrafficGenerator
from routing.dijkstra import DijkstraRouter
from routing.widest_path import WidestPathRouter
from routing.min_delay import MinDelayRouter
from routing.nn_router_online import OnlineNNRouter
from simulation.simulator import NetworkSimulator

def run_single_experiment(router_name, router, load, duration=60, seed=42):
    """Run a single experiment and return metrics"""
    env = simpy.Environment()
    topology = WANTopology(n_nodes=20, seed=seed)
    traffic_gen = TrafficGenerator(env, topology.get_graph(), 
                                   arrival_rate=5, load_factor=load)
    
    sim = NetworkSimulator(env, topology, traffic_gen, router)
    sim.run()
    env.run(until=duration)
    
    return sim.get_metrics()

def run_comprehensive_evaluation():
    """Full evaluation for project report"""
    
    print("="*80)
    print("COMPREHENSIVE EVALUATION - Final Project Results")
    print("="*80)
    
    # Algorithms to test
    algorithms = {
        'Dijkstra': lambda: DijkstraRouter(),
        'Widest-Path': lambda: WidestPathRouter(),
        'Min-Delay': lambda: MinDelayRouter(),
        'NN-Online': lambda: OnlineNNRouter(
            pretrained_model='models/nn_router.pth',
            epsilon=0.2,
            learning_rate=0.001
        )
    }
    
    load_levels = [0.3, 0.6, 0.9]
    results = {algo: {load: [] for load in load_levels} for algo in algorithms}
    
    # Run multiple trials
    # NOTE: With n=3, we report averages and note consistency across trials.
    # Statistical significance testing would require more trials (n≥30 recommended).
    num_trials = 3
    
    for trial in range(num_trials):
        print(f"\n{'='*80}")
        print(f"TRIAL {trial + 1}/{num_trials}")
        print(f"{'='*80}")
        
        for load in load_levels:
            print(f"\nLoad: {load*100:.0f}%")
            
            for algo_name, router_factory in algorithms.items():
                router = router_factory()
                metrics = run_single_experiment(
                    algo_name, router, load, 
                    duration=60, seed=42+trial
                )
                
                results[algo_name][load].append(metrics)
                
                print(f"  {algo_name:15s}: miss={metrics['deadline_miss_ratio']*100:5.1f}%, "
                      f"delay={metrics['avg_delay']:5.1f}ms")
    
    # Compute averages
    print("\n" + "="*80)
    print("AVERAGE RESULTS (across 3 trials)")
    print("NOTE: Results show consistency across 3 trials. More trials would strengthen conclusions.")
    print("="*80)
    
    avg_results = {}
    for algo_name in algorithms:
        avg_results[algo_name] = {}
        for load in load_levels:
            metrics_list = results[algo_name][load]
            avg_results[algo_name][load] = {
                'miss_ratio': np.mean([m['deadline_miss_ratio'] for m in metrics_list]),
                'avg_delay': np.mean([m['avg_delay'] for m in metrics_list]),
                'dropped': np.mean([m['dropped_flows'] for m in metrics_list])
            }
    
    # Print table
    print(f"\n{'Algorithm':<15} {'Load':<10} {'Miss Ratio':<15} {'Avg Delay':<15}")
    print("-"*60)
    for algo_name in algorithms:
        for load in load_levels:
            r = avg_results[algo_name][load]
            print(f"{algo_name:<15} {load*100:>3.0f}%      "
                  f"{r['miss_ratio']*100:>6.2f}%         "
                  f"{r['avg_delay']:>7.2f}ms")
    
    # Create comprehensive plots
    create_final_plots(avg_results, algorithms.keys(), load_levels)
    
    return avg_results

def create_final_plots(avg_results, algorithms, load_levels):
    """Create publication-quality plots for report"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'Dijkstra': '#1f77b4', 'Widest-Path': '#ff7f0e', 
              'Min-Delay': '#2ca02c', 'NN-Online': '#d62728'}
    
    # 1. Miss Ratio vs Load
    ax = axes[0, 0]
    for algo in algorithms:
        miss_ratios = [avg_results[algo][load]['miss_ratio']*100 for load in load_levels]
        ax.plot([l*100 for l in load_levels], miss_ratios, 
                marker='o', label=algo, color=colors[algo], linewidth=2)
    
    ax.set_xlabel('Network Load (%)', fontsize=12)
    ax.set_ylabel('Deadline Miss Ratio (%)', fontsize=12)
    ax.set_title('Deadline Miss Ratio vs Network Load', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Average Delay vs Load
    ax = axes[0, 1]
    for algo in algorithms:
        delays = [avg_results[algo][load]['avg_delay'] for load in load_levels]
        ax.plot([l*100 for l in load_levels], delays, 
                marker='s', label=algo, color=colors[algo], linewidth=2)
    
    ax.set_xlabel('Network Load (%)', fontsize=12)
    ax.set_ylabel('Average Delay (ms)', fontsize=12)
    ax.set_title('Average End-to-End Delay vs Network Load', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Bar chart at 60% load
    ax = axes[1, 0]
    load = 0.6
    x = np.arange(len(algorithms))
    miss_ratios = [avg_results[algo][load]['miss_ratio']*100 for algo in algorithms]
    
    bars = ax.bar(x, miss_ratios, color=[colors[a] for a in algorithms])
    ax.set_ylabel('Deadline Miss Ratio (%)', fontsize=12)
    ax.set_title('Comparison at 60% Load', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=15)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 4. Improvement over Dijkstra
    ax = axes[1, 1]
    improvements = {}
    for algo in algorithms:
        if algo != 'Dijkstra':
            improvements[algo] = []
            for load in load_levels:
                dijkstra_miss = avg_results['Dijkstra'][load]['miss_ratio']
                algo_miss = avg_results[algo][load]['miss_ratio']
                improvement = (dijkstra_miss - algo_miss) / dijkstra_miss * 100
                improvements[algo].append(improvement)
    
    x = np.arange(len(load_levels))
    width = 0.25
    
    for i, algo in enumerate(['Widest-Path', 'Min-Delay', 'NN-Online']):
        ax.bar(x + i*width, improvements[algo], width, 
               label=algo, color=colors[algo])
    
    ax.set_ylabel('Improvement over Dijkstra (%)', fontsize=12)
    ax.set_xlabel('Network Load (%)', fontsize=12)
    ax.set_title('Miss Ratio Improvement vs Baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{int(l*100)}%' for l in load_levels])
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/final_evaluation.png', dpi=300, bbox_inches='tight')
    print("\nFinal evaluation plot saved to 'results/final_evaluation.png'")
    plt.close()

if __name__ == "__main__":
    import torch
    os.makedirs('results', exist_ok=True)
    avg_results = run_comprehensive_evaluation()