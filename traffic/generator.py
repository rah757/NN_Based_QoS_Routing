# traffic/generator.py

import random
import numpy as np
from traffic.flow import Flow, TrafficClass

class TrafficGenerator:
    """Generates realistic real-time traffic using Poisson arrival process"""
    
    def __init__(self, env, network, arrival_rate=10, load_factor=0.6):
        """
        Args:
            env: SimPy environment
            network: NetworkX graph
            arrival_rate: Average flows per second (lambda for Poisson)
            load_factor: Network load (0.3=light, 0.6=medium, 0.9=heavy)
        """
        self.env = env
        self.network = network
        self.arrival_rate = arrival_rate * load_factor
        self.load_factor = load_factor
        
        # Traffic class distribution (probabilities)
        self.traffic_mix = {
            TrafficClass.VIDEO: 0.5,      # 50% video
            TrafficClass.VOIP: 0.3,       # 30% VoIP
            TrafficClass.INDUSTRIAL: 0.2  # 20% industrial
        }
        
        # Tracking
        self.flow_counter = 0
        self.generated_flows = []
        
    def generate_traffic(self):
        """Generator function for SimPy process"""
        while True:
            # Poisson inter-arrival time
            inter_arrival = random.expovariate(self.arrival_rate)
            yield self.env.timeout(inter_arrival)
            
            # Create new flow
            flow = self._create_flow()
            self.generated_flows.append(flow)
            
            print(f"[{self.env.now:.3f}s] Generated: {flow}")
    
    def _create_flow(self):
        """Create a single flow with random parameters"""
        # Select traffic class based on distribution
        traffic_class = random.choices(
            list(self.traffic_mix.keys()),
            weights=list(self.traffic_mix.values())
        )[0]
        
        # Random source and destination
        nodes = list(self.network.nodes())
        source = random.choice(nodes)
        destination = random.choice([n for n in nodes if n != source])
        
        # Flow size based on traffic class
        if traffic_class == TrafficClass.VIDEO:
            size_bytes = random.randint(10000, 50000)  # 10-50 KB
        elif traffic_class == TrafficClass.VOIP:
            size_bytes = random.randint(160, 320)      # 160-320 bytes
        else:  # INDUSTRIAL
            size_bytes = random.randint(64, 256)       # 64-256 bytes
        
        flow = Flow(
            flow_id=self.flow_counter,
            source=source,
            destination=destination,
            traffic_class=traffic_class,
            size_bytes=size_bytes,
            creation_time=self.env.now
        )
        
        self.flow_counter += 1
        return flow
    
    def get_statistics(self):
        """Get traffic generation statistics"""
        total = len(self.generated_flows)
        if total == 0:
            return {}
        
        stats = {
            'total_flows': total,
            'by_class': {
                tc: sum(1 for f in self.generated_flows if f.traffic_class == tc)
                for tc in TrafficClass
            },
            'avg_size': np.mean([f.size_bytes for f in self.generated_flows]),
            'generation_rate': total / self.env.now if self.env.now > 0 else 0
        }
        return stats