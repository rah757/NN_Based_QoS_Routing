# extract_delay_values.py - Extract avg_delay for Table II

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simpy
import numpy as np
from network.topology import WANTopology
from traffic.generator import TrafficGenerator
from routing.dijkstra import DijkstraRouter
from routing.min_delay import MinDelayRouter
from routing.nn_router import NNRouter
from simulation.simulator import NetworkSimulator

def extract_avg_delay():
    """Extract average delay values for Table II"""
    
    print("="*80)
    print("EXTRACTING AVERAGE DELAY VALUES FOR TABLE II")
    print("="*80)
    
    algorithms = {
        'Dijkstra': lambda: DijkstraRouter(),
        'Min-Delay': lambda: MinDelayRouter(),
        'NN-Supervised': lambda: NNRouter(pretrained_model='models/nn_router_supervised.pth', epsilon=0.0)
    }
    
    results = {}
    
    # Run 3 trials per condition to assess consistency
    # NOTE: With n=3, we report averages and note consistency across trials.
    # Statistical significance testing would require more trials (n≥30 recommended).
    for load in [0.3, 0.6, 0.9]:
        print(f"\n{'='*80}")
        print(f"LOAD: {load*100:.0f}%")
        print(f"{'='*80}\n")
        
        results[load] = {}
        
        for name, router_factory in algorithms.items():
            all_metrics = []
            
            for trial in range(3):
                print(f"  {name} - Trial {trial+1}...", end=" ")
                
                env = simpy.Environment()
                topology = WANTopology(n_nodes=20, seed=42+trial)
                router = router_factory()
                traffic_gen = TrafficGenerator(env, topology.get_graph(), 
                                              arrival_rate=5, load_factor=load)
                
                sim = NetworkSimulator(env, topology, traffic_gen, router)
                sim.run()
                env.run(until=60)
                
                metrics = sim.get_metrics()
                all_metrics.append(metrics)
                
                print(f"avg_delay={metrics['avg_delay']:.2f}ms, "
                      f"admission={metrics['admission_ratio']*100:.1f}%, "
                      f"miss={metrics['deadline_miss_ratio']*100:.1f}%, "
                      f"wcrt={metrics['wcrt']:.1f}ms")
            
            # Average across 3 trials
            avg_metrics = {
                'avg_delay': np.mean([m['avg_delay'] for m in all_metrics]),
                'admission_rate': np.mean([m['admission_ratio'] for m in all_metrics]) * 100,
                'miss_ratio': np.mean([m['deadline_miss_ratio'] for m in all_metrics]) * 100,
                'guarantee': np.mean([1 - m['deadline_miss_ratio'] for m in all_metrics]) * 100,
                'wcrt': np.mean([m['wcrt'] for m in all_metrics])
            }
            results[load][name] = avg_metrics
            
            print(f"  {name} - Average: delay={avg_metrics['avg_delay']:.2f}ms, "
                  f"admission={avg_metrics['admission_rate']:.1f}%, "
                  f"guarantee={avg_metrics['guarantee']:.1f}%, "
                  f"miss={avg_metrics['miss_ratio']:.1f}%, "
                  f"wcrt={avg_metrics['wcrt']:.1f}ms\n")
    
    # Print complete Table II format
    print("\n" + "="*80)
    print("COMPLETE TABLE II - ALL METRICS (3-Trial Averages)")
    print("NOTE: Results show consistency across 3 trials. More trials would strengthen conclusions.")
    print("="*80)
    
    # Average Delay
    print(f"\n{'Algorithm':<20} {'30% Load':<15} {'60% Load':<15} {'90% Load':<15}")
    print("AVERAGE DELAY (ms):")
    print("-"*80)
    for name in ['Dijkstra', 'Min-Delay', 'NN-Supervised']:
        row = f"{name:<20}"
        for load in [0.3, 0.6, 0.9]:
            delay = results[load][name]['avg_delay']
            row += f"{delay:>8.1f}ms       "
        print(row)
    
    # Admission Rate
    print(f"\n{'Algorithm':<20} {'30% Load':<15} {'60% Load':<15} {'90% Load':<15}")
    print("ADMISSION RATE (%):")
    print("-"*80)
    for name in ['Dijkstra', 'Min-Delay', 'NN-Supervised']:
        row = f"{name:<20}"
        for load in [0.3, 0.6, 0.9]:
            admission = results[load][name]['admission_rate']
            row += f"{admission:>8.1f}%       "
        print(row)
    
    # Deadline Guarantee
    print(f"\n{'Algorithm':<20} {'30% Load':<15} {'60% Load':<15} {'90% Load':<15}")
    print("DEADLINE GUARANTEE (%):")
    print("-"*80)
    for name in ['Dijkstra', 'Min-Delay', 'NN-Supervised']:
        row = f"{name:<20}"
        for load in [0.3, 0.6, 0.9]:
            guarantee = results[load][name]['guarantee']
            row += f"{guarantee:>8.1f}%       "
        print(row)
    
    # Miss Ratio
    print(f"\n{'Algorithm':<20} {'30% Load':<15} {'60% Load':<15} {'90% Load':<15}")
    print("MISS RATIO (%):")
    print("-"*80)
    for name in ['Dijkstra', 'Min-Delay', 'NN-Supervised']:
        row = f"{name:<20}"
        for load in [0.3, 0.6, 0.9]:
            miss = results[load][name]['miss_ratio']
            row += f"{miss:>8.1f}%       "
        print(row)
    
    # WCRT
    print(f"\n{'Algorithm':<20} {'30% Load':<15} {'60% Load':<15} {'90% Load':<15}")
    print("WCRT (ms):")
    print("-"*80)
    for name in ['Dijkstra', 'Min-Delay', 'NN-Supervised']:
        row = f"{name:<20}"
        for load in [0.3, 0.6, 0.9]:
            wcrt = results[load][name]['wcrt']
            row += f"{wcrt:>8.1f}ms       "
        print(row)
    
    # Comparison with user's existing values
    print("\n" + "="*80)
    print("COMPARISON WITH YOUR EXISTING VALUES (90% Load)")
    print("="*80)
    print(f"{'Metric':<20} {'Dijkstra (Your)':<20} {'Dijkstra (New)':<20} {'Match?':<10}")
    print("-"*80)
    user_dijkstra_90 = {'admission': 91.2, 'guarantee': 81.7, 'miss': 18.3, 'wcrt': 164.8}
    new_dijkstra_90 = results[0.9]['Dijkstra']
    print(f"{'Admission':<20} {user_dijkstra_90['admission']:<20.1f} {new_dijkstra_90['admission_rate']:<20.1f} "
          f"{'✓' if abs(user_dijkstra_90['admission'] - new_dijkstra_90['admission_rate']) < 2 else '✗'}")
    print(f"{'Guarantee':<20} {user_dijkstra_90['guarantee']:<20.1f} {new_dijkstra_90['guarantee']:<20.1f} "
          f"{'✓' if abs(user_dijkstra_90['guarantee'] - new_dijkstra_90['guarantee']) < 2 else '✗'}")
    print(f"{'Miss Ratio':<20} {user_dijkstra_90['miss']:<20.1f} {new_dijkstra_90['miss_ratio']:<20.1f} "
          f"{'✓' if abs(user_dijkstra_90['miss'] - new_dijkstra_90['miss_ratio']) < 2 else '✗'}")
    print(f"{'WCRT':<20} {user_dijkstra_90['wcrt']:<20.1f} {new_dijkstra_90['wcrt']:<20.1f} "
          f"{'✓' if abs(user_dijkstra_90['wcrt'] - new_dijkstra_90['wcrt']) < 5 else '✗'}")
    
    print(f"\n{'Metric':<20} {'Min-Delay (Your)':<20} {'Min-Delay (New)':<20} {'Match?':<10}")
    print("-"*80)
    user_min_90 = {'admission': 94.0, 'guarantee': 78.0, 'miss': 22.0, 'wcrt': 158.7}
    new_min_90 = results[0.9]['Min-Delay']
    print(f"{'Admission':<20} {user_min_90['admission']:<20.1f} {new_min_90['admission_rate']:<20.1f} "
          f"{'✓' if abs(user_min_90['admission'] - new_min_90['admission_rate']) < 2 else '✗'}")
    print(f"{'Guarantee':<20} {user_min_90['guarantee']:<20.1f} {new_min_90['guarantee']:<20.1f} "
          f"{'✓' if abs(user_min_90['guarantee'] - new_min_90['guarantee']) < 2 else '✗'}")
    print(f"{'Miss Ratio':<20} {user_min_90['miss']:<20.1f} {new_min_90['miss_ratio']:<20.1f} "
          f"{'✓' if abs(user_min_90['miss'] - new_min_90['miss_ratio']) < 2 else '✗'}")
    print(f"{'WCRT':<20} {user_min_90['wcrt']:<20.1f} {new_min_90['wcrt']:<20.1f} "
          f"{'✓' if abs(user_min_90['wcrt'] - new_min_90['wcrt']) < 5 else '✗'}")
    
    print(f"\n{'Metric':<20} {'NN-Supervised (Your)':<20} {'NN-Supervised (New)':<20} {'Match?':<10}")
    print("-"*80)
    user_nn_90 = {'admission': 89.7, 'guarantee': 83.0, 'miss': 17.0, 'wcrt': 171.8}
    new_nn_90 = results[0.9]['NN-Supervised']
    print(f"{'Admission':<20} {user_nn_90['admission']:<20.1f} {new_nn_90['admission_rate']:<20.1f} "
          f"{'✓' if abs(user_nn_90['admission'] - new_nn_90['admission_rate']) < 2 else '✗'}")
    print(f"{'Guarantee':<20} {user_nn_90['guarantee']:<20.1f} {new_nn_90['guarantee']:<20.1f} "
          f"{'✓' if abs(user_nn_90['guarantee'] - new_nn_90['guarantee']) < 2 else '✗'}")
    print(f"{'Miss Ratio':<20} {user_nn_90['miss']:<20.1f} {new_nn_90['miss_ratio']:<20.1f} "
          f"{'✓' if abs(user_nn_90['miss'] - new_nn_90['miss_ratio']) < 2 else '✗'}")
    print(f"{'WCRT':<20} {user_nn_90['wcrt']:<20.1f} {new_nn_90['wcrt']:<20.1f} "
          f"{'✓' if abs(user_nn_90['wcrt'] - new_nn_90['wcrt']) < 5 else '✗'}")
    
    return results

if __name__ == "__main__":
    results = extract_avg_delay()

