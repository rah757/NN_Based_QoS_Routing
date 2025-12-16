# experiments/final_results.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import simpy
import numpy as np
import matplotlib.pyplot as plt
from network.topology import WANTopology
from traffic.generator import TrafficGenerator
from routing.dijkstra import DijkstraRouter
from routing.min_delay import MinDelayRouter
from routing.nn_router import NNRouter
from simulation.simulator import NetworkSimulator
from traffic.flow import TrafficClass

def final_comprehensive_results():
    """Generate final results for report submission"""
    
    print("="*80)
    print("FINAL PROJECT RESULTS - Neural Network QoS Routing")
    print("="*80)
    print("\nNOTE: Evaluation uses load levels [30%, 60%, 90%].")
    print("      Training data was collected at [30%, 50%, 70%, 90%] for better generalization.")
    print("      Evaluating at 60% (unseen during training) tests the model's generalization ability.\n")
    
    algorithms = {
        'Dijkstra': lambda: DijkstraRouter(),
        'Min-Delay': lambda: MinDelayRouter(),
        'NN-Supervised': lambda: NNRouter(pretrained_model='models/nn_router_supervised.pth', epsilon=0.0)
    }
    
    results = {}
    detailed_results = {}
    
    # Run 3 trials per condition to assess consistency
    # NOTE: With n=3, we report averages and note consistency across trials.
    # Statistical significance testing would require more trials (n≥30 recommended).
    # Evaluation at 30/60/90 provides good spread for reporting.
    # Training used 30/50/70/90 to expose model to more diverse conditions.
    # Evaluating at 60% (unseen) demonstrates generalization.
    for load in [0.3, 0.6, 0.9]:
        print(f"\n{'='*80}")
        print(f"LOAD: {load*100:.0f}% (Arrival Rate: {5*load:.1f} flows/sec)")
        print(f"{'='*80}\n")
        
        results[load] = {}
        detailed_results[load] = {}
        
        for name, router_factory in algorithms.items():
            trials = []
            
            for trial in range(3):
                env = simpy.Environment()
                topology = WANTopology(n_nodes=20, seed=42+trial)
                router = router_factory()
                traffic_gen = TrafficGenerator(env, topology.get_graph(), 
                                              arrival_rate=5, load_factor=load)
                
                sim = NetworkSimulator(env, topology, traffic_gen, router)
                sim.run()
                env.run(until=60)
                
                metrics = sim.get_metrics()
                trials.append(metrics)
            
            # Average across trials
            avg_metrics = {
                'total_flows': np.mean([m['total_flows'] for m in trials]),
                'completed': np.mean([m['completed_flows'] for m in trials]),
                'rejected': np.mean([m['rejected_flows'] for m in trials]),
                'admission_rate': np.mean([m['admission_ratio'] for m in trials]) * 100,
                'miss_ratio': np.mean([m['deadline_miss_ratio'] for m in trials]) * 100,
                'guarantee': np.mean([(1-m['deadline_miss_ratio']) for m in trials]) * 100,
                'avg_delay': np.mean([m['avg_delay'] for m in trials]),
                'wcrt': np.mean([m['wcrt'] for m in trials])
            }
            
            results[load][name] = avg_metrics
            detailed_results[load][name] = trials
            
            print(f"{name}:")
            print(f"  Admission Rate: {avg_metrics['admission_rate']:.1f}%")
            print(f"  Deadline Guarantee: {avg_metrics['guarantee']:.1f}%")
            print(f"  Miss Ratio: {avg_metrics['miss_ratio']:.1f}%")
            print(f"  WCRT: {avg_metrics['wcrt']:.1f}ms")
            print(f"  Avg Delay: {avg_metrics['avg_delay']:.1f}ms")
            print()
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE - 3-Trial Averages")
    print("NOTE: Results show consistency across 3 trials. More trials would strengthen conclusions.")
    print("="*80)
    print(f"{'Algorithm':<15} {'Load':<8} {'Admission':<12} {'Guarantee':<12} {'WCRT':<12} {'Miss Ratio':<12}")
    print("-"*80)
    
    for name in algorithms.keys():
        for load in [0.3, 0.6, 0.9]:
            r = results[load][name]
            print(f"{name:<15} {int(load*100):>3}%     "
                  f"{r['admission_rate']:>6.1f}%      "
                  f"{r['guarantee']:>6.1f}%      "
                  f"{r['wcrt']:>7.1f}ms     "
                  f"{r['miss_ratio']:>6.1f}%")
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Best WCRT at 90% load
    wcrt_90 = {name: results[0.9][name]['wcrt'] for name in algorithms.keys()}
    best_wcrt_algo = min(wcrt_90, key=wcrt_90.get)
    best_wcrt = wcrt_90[best_wcrt_algo]
    dijkstra_wcrt = wcrt_90['Dijkstra']
    
    print(f"\n1. WORST-CASE RESPONSE TIME (90% Load):")
    print(f"   {best_wcrt_algo}: {best_wcrt:.1f}ms")
    print(f"   Dijkstra: {dijkstra_wcrt:.1f}ms")
    print(f"   Improvement: {dijkstra_wcrt - best_wcrt:.1f}ms ({(dijkstra_wcrt-best_wcrt)/dijkstra_wcrt*100:.1f}%)")
    
    # Deadline guarantee at 90%
    guarantee_90 = {name: results[0.9][name]['guarantee'] for name in algorithms.keys()}
    best_guarantee_algo = max(guarantee_90, key=guarantee_90.get)
    
    print(f"\n2. DEADLINE GUARANTEE (90% Load):")
    print(f"   {best_guarantee_algo}: {guarantee_90[best_guarantee_algo]:.1f}%")
    print(f"   Dijkstra: {guarantee_90['Dijkstra']:.1f}%")
    
    # Admission vs guarantee trade-off
    nn_admission = results[0.9]['NN-Supervised']['admission_rate']
    nn_guarantee = results[0.9]['NN-Supervised']['guarantee']
    min_admission = results[0.9]['Min-Delay']['admission_rate']
    min_guarantee = results[0.9]['Min-Delay']['guarantee']
    
    print(f"\n3. ADMISSION vs GUARANTEE TRADE-OFF (90% Load):")
    print(f"   Min-Delay: {min_admission:.1f}% admission, {min_guarantee:.1f}% guarantee")
    print(f"   NN-Supervised: {nn_admission:.1f}% admission, {nn_guarantee:.1f}% guarantee")
    print(f"   NN trades {min_admission-nn_admission:.1f}% admission for {nn_guarantee-min_guarantee:.1f}% better guarantee")
    
    # Create final plot
    create_final_plot(results, algorithms.keys())
    
    return results

def create_final_plot(results, algorithms):
    """Create publication-quality final plot"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Neural Network QoS Routing - Final Results', fontsize=16, fontweight='bold')
    
    colors = {'Dijkstra': '#1f77b4', 'Min-Delay': '#2ca02c', 'NN-Supervised': '#d62728'}
    loads = [0.3, 0.6, 0.9]
    
    # 1. Deadline Miss Ratio
    ax = axes[0, 0]
    for algo in algorithms:
        miss_ratios = [results[load][algo]['miss_ratio'] for load in loads]
        ax.plot([l*100 for l in loads], miss_ratios, 
                marker='o', label=algo, color=colors[algo], linewidth=2.5, markersize=8)
    ax.set_xlabel('Network Load (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Deadline Miss Ratio (%)', fontsize=12, fontweight='bold')
    ax.set_title('Deadline Miss Ratio vs Load', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. WCRT
    ax = axes[0, 1]
    for algo in algorithms:
        wcrts = [results[load][algo]['wcrt'] for load in loads]
        ax.plot([l*100 for l in loads], wcrts, 
                marker='s', label=algo, color=colors[algo], linewidth=2.5, markersize=8)
    ax.set_xlabel('Network Load (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Worst-Case Response Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('WCRT vs Load', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 3. Admission Rate
    ax = axes[1, 0]
    for algo in algorithms:
        admission = [results[load][algo]['admission_rate'] for load in loads]
        ax.plot([l*100 for l in loads], admission, 
                marker='^', label=algo, color=colors[algo], linewidth=2.5, markersize=8)
    ax.set_xlabel('Network Load (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Admission Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Admission Rate vs Load', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 4. Bar chart at 90% load
    ax = axes[1, 1]
    x = np.arange(len(algorithms))
    width = 0.35
    
    guarantee_90 = [results[0.9][algo]['guarantee'] for algo in algorithms]
    wcrt_90 = [results[0.9][algo]['wcrt'] for algo in algorithms]
    
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x - width/2, guarantee_90, width, label='Deadline Guarantee', 
                   color=[colors[a] for a in algorithms], alpha=0.8)
    bars2 = ax2.bar(x + width/2, wcrt_90, width, label='WCRT', 
                    color=[colors[a] for a in algorithms], alpha=0.5, hatch='//')
    
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Deadline Guarantee (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('WCRT (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Performance at 90% Load', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/final_project_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 Final plot saved to 'results/final_project_results.png'")
    plt.close()

if __name__ == "__main__":
    results = final_comprehensive_results()