# simulation/metrics.py

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class MetricsCollector:
    """Collect and visualize simulation metrics"""
    
    def __init__(self):
        self.results = defaultdict(list)
    
    def add_run(self, algorithm_name, metrics):
        """Add results from a simulation run"""
        self.results[algorithm_name].append(metrics)
    
    def compare_algorithms(self):
        """Print comparison table"""
        print("\n" + "="*80)
        print("ALGORITHM COMPARISON")
        print("="*80)
        print(f"{'Algorithm':<20} {'Miss Ratio':<15} {'Avg Delay':<15} {'Dropped':<10}")
        print("-"*80)
        
        for algo, runs in self.results.items():
            avg_metrics = self._average_metrics(runs)
            print(f"{algo:<20} {avg_metrics['deadline_miss_ratio']*100:>6.2f}%        "
                  f"{avg_metrics['avg_delay']:>7.2f}ms       "
                  f"{avg_metrics['dropped_flows']:>6.1f}")
    
    def _average_metrics(self, runs):
        """Average metrics across multiple runs"""
        avg = {}
        keys = ['deadline_miss_ratio', 'avg_delay', 'dropped_flows', 'completed_flows']
        for key in keys:
            avg[key] = np.mean([r[key] for r in runs])
        return avg
    
    def plot_comparison(self, save_path='comparison.png'):
        """Create comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        algorithms = list(self.results.keys())
        
        # 1. Deadline Miss Ratio
        ax = axes[0, 0]
        miss_ratios = [self._average_metrics(self.results[algo])['deadline_miss_ratio'] * 100 
                       for algo in algorithms]
        ax.bar(algorithms, miss_ratios, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(algorithms)])
        ax.set_ylabel('Deadline Miss Ratio (%)')
        ax.set_title('Deadline Miss Ratio by Algorithm')
        ax.set_ylim(0, max(miss_ratios) * 1.2)
        
        # 2. Average Delay
        ax = axes[0, 1]
        avg_delays = [self._average_metrics(self.results[algo])['avg_delay'] 
                      for algo in algorithms]
        ax.bar(algorithms, avg_delays, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(algorithms)])
        ax.set_ylabel('Average Delay (ms)')
        ax.set_title('Average End-to-End Delay')
        
        # 3. Per-class miss ratio
        ax = axes[1, 0]
        from traffic.flow import TrafficClass
        x = np.arange(len(TrafficClass))
        width = 0.8 / len(algorithms)
        
        for i, algo in enumerate(algorithms):
            runs = self.results[algo]
            class_miss = []
            for tc in TrafficClass:
                tc_metrics = [r['by_class'].get(tc.value, {'miss_ratio': 0})['miss_ratio'] 
                             for r in runs if tc.value in r['by_class']]
                class_miss.append(np.mean(tc_metrics) * 100 if tc_metrics else 0)
            
            ax.bar(x + i*width, class_miss, width, label=algo)
        
        ax.set_ylabel('Miss Ratio (%)')
        ax.set_title('Miss Ratio by Traffic Class')
        ax.set_xticks(x + width * (len(algorithms)-1) / 2)
        ax.set_xticklabels([tc.value for tc in TrafficClass])
        ax.legend()
        
        # 4. Completion rate
        ax = axes[1, 1]
        completed = [self._average_metrics(self.results[algo])['completed_flows'] 
                    for algo in algorithms]
        ax.bar(algorithms, completed, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(algorithms)])
        ax.set_ylabel('Completed Flows')
        ax.set_title('Throughput (Completed Flows)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\nComparison plot saved to '{save_path}'")
        plt.close()