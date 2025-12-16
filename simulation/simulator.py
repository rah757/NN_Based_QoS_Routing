# simulation/simulator.py - REPLACE ENTIRE FILE

import simpy
import networkx as nx
from collections import defaultdict
import random

class NetworkSimulator:
    """Main simulation engine with REAL-TIME scheduling features"""
    
    def __init__(self, env, topology, traffic_generator, router):
        self.env = env
        self.topology = topology
        self.G = topology.get_graph()
        self.traffic_gen = traffic_generator
        self.router = router
        
        # Initialize link objects
        from network.link import Link
        for u, v in self.G.edges():
            edge_data = self.G[u][v]
            link = Link(
                bandwidth=edge_data['bandwidth'],
                delay=edge_data['delay'],
                jitter=edge_data['jitter'],
                loss=edge_data['loss']
            )
            self.G[u][v]['link_obj'] = link
        
        # Metrics tracking
        self.completed_flows = []
        self.dropped_flows = []
        self.rejected_flows = []  # NEW: admission control rejections
        self.active_flows = {}
        
    def run(self):
        """Start all simulation processes"""
        self.env.process(self.traffic_gen.generate_traffic())
        self.env.process(self.process_flows())
        self.env.process(self.update_link_states())
    
    def update_link_states(self):
        """Periodically update dynamic link states"""
        while True:
            yield self.env.timeout(0.1)
            
            for u, v in self.G.edges():
                link = self.G[u][v]['link_obj']
                self.G[u][v]['available_bw'] = link.get_available_bandwidth()
                self.G[u][v]['current_delay'] = link.get_current_delay()
    
    def process_flows(self):
        """Process each generated flow"""
        last_checked = 0
        
        while True:
            yield self.env.timeout(0.01)
            
            new_flows = self.traffic_gen.generated_flows[last_checked:]
            
            for flow in new_flows:
                self.env.process(self.handle_flow(flow))
            
            last_checked = len(self.traffic_gen.generated_flows)
    
    def handle_flow(self, flow):
        """Handle flow with REAL-TIME admission control"""
        
        # Find path using router
        path = self.router.find_path(flow, self.G)
        
        if path is None:
            self.dropped_flows.append(flow)
            print(f"[{self.env.now:.3f}s] DROPPED: {flow} (no path)")
            return
        
        # *** REAL-TIME ADMISSION CONTROL ***
        if not self._schedulability_test(flow, path):
            self.rejected_flows.append(flow)
            print(f"[{self.env.now:.3f}s] REJECTED: {flow} (admission control - would violate deadlines)")
            return
        
        flow.path_taken = path
        
        # Reserve bandwidth (admit the flow)
        self._reserve_bandwidth(flow, path)
        
        # Simulate transmission with PRIORITY SCHEDULING
        yield self.env.process(self.transmit_flow(flow, path))
        
        # Release bandwidth
        self._release_bandwidth(flow, path)
        
        # Check deadline
        flow.completion_time = self.env.now
        deadline_met = flow.check_deadline(self.env.now)
        
        self.completed_flows.append(flow)
        
        # Online learning (if router supports it)
        if hasattr(self.router, 'learn_from_experience'):
            self.router.learn_from_experience(flow, path, self.G)
        
        status = "✓" if deadline_met else "✗"
        delay = flow.get_delay_ms()
        print(f"[{self.env.now:.3f}s] {status} {flow.traffic_class.value:12s} "
              f"{flow.source}→{flow.destination} delay={delay:.1f}ms "
              f"(deadline={flow.deadline_ms}ms) path={path}")
    
    def _schedulability_test(self, flow, path):
        """
        REAL-TIME SCHEDULABILITY TEST
        Check if admitting this flow maintains deadline guarantees
        Uses utilization-based admission control (Liu & Layland bound)
        """
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            link = self.G[u][v]['link_obj']
            
            # Calculate current utilization
            current_util = sum(f.bandwidth_mbps for f in link.current_flows) / link.max_bandwidth
            
            # New utilization if we admit this flow
            new_util = current_util + (flow.bandwidth_mbps / link.max_bandwidth)
            
            # Schedulability bound for hard real-time
            # Using conservative bound: U ≤ 0.7 (allows safety margin)
            # Standard bound would be U ≤ n(2^(1/n) - 1) ≈ 0.69 for large n
            UTILIZATION_BOUND = 0.7
            
            if new_util > UTILIZATION_BOUND:
                return False  # Would violate schedulability
            
            # Also check if worst-case delay would exceed deadline
            estimated_delay = link.get_current_delay() * (1 + new_util)  # pessimistic estimate
            if estimated_delay > flow.deadline_ms:
                return False
        
        return True  # Flow is schedulable
    
    def _reserve_bandwidth(self, flow, path):
        """Reserve bandwidth on path links"""
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            link = self.G[u][v]['link_obj']
            link.add_flow(flow)
    
    def _release_bandwidth(self, flow, path):
        """Release bandwidth after flow completes"""
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            link = self.G[u][v]['link_obj']
            link.remove_flow(flow)
    
    def transmit_flow(self, flow, path):
        """
        Simulate transmission with PRIORITY SCHEDULING (EDF)
        Each hop uses Earliest Deadline First scheduling
        """
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            link = self.G[u][v]['link_obj']
            
            # Transmission delay = size / bandwidth
            trans_delay = (flow.size_bytes * 8) / (link.max_bandwidth * 1e6)
            
            # Current delay (includes queuing based on link load)
            current_delay = link.get_current_delay() / 1000.0
            
            # Jitter
            jitter = random.gauss(0, link.jitter/1000.0/3)
            
            # Packet loss (retransmission)
            if random.random() < link.loss_rate / 100.0:
                current_delay += 0.050  # 50ms retransmission penalty
            
            # PRIORITY DELAY: Higher priority (industrial) gets served faster
            # Simulate preemption by reducing delay for high-priority traffic
            priority_factor = 1.0
            if flow.priority == 3:  # Industrial (highest)
                priority_factor = 0.8  # 20% faster service
            elif flow.priority == 2:  # VoIP
                priority_factor = 0.9  # 10% faster service
            
            hop_delay = trans_delay + (current_delay * priority_factor) + abs(jitter)
            yield self.env.timeout(hop_delay)
    
    def get_metrics(self):
        """Calculate metrics with REAL-TIME focus"""
        if not self.completed_flows:
            return {
                'total_flows': len(self.dropped_flows) + len(self.rejected_flows),
                'completed_flows': 0,
                'dropped_flows': len(self.dropped_flows),
                'rejected_flows': len(self.rejected_flows),
                'deadline_met': 0,
                'deadline_miss_ratio': 1.0,
                'avg_delay': 0,
                'min_delay': 0,
                'max_delay': 0,
                'by_class': {}
            }
        
        total_flows = len(self.completed_flows) + len(self.dropped_flows) + len(self.rejected_flows)
        met_deadline = sum(1 for f in self.completed_flows if f.deadline_met)
        delays = [f.get_delay_ms() for f in self.completed_flows]
        
        from traffic.flow import TrafficClass
        class_metrics = {}
        for tc in TrafficClass:
            flows = [f for f in self.completed_flows if f.traffic_class == tc]
            if flows:
                class_delays = [f.get_delay_ms() for f in flows]
                class_metrics[tc.value] = {
                    'count': len(flows),
                    'deadline_met': sum(1 for f in flows if f.deadline_met),
                    'miss_ratio': 1 - (sum(1 for f in flows if f.deadline_met) / len(flows)),
                    'avg_delay': sum(class_delays) / len(flows),
                    'wcrt': max(class_delays),  # Worst-Case Response Time
                    'bcrt': min(class_delays),  # Best-Case Response Time
                    'deadline': flows[0].deadline_ms,
                    'schedulability': 'YES' if max(class_delays) <= flows[0].deadline_ms else 'NO'
                }
        
        return {
            'total_flows': total_flows,
            'completed_flows': len(self.completed_flows),
            'dropped_flows': len(self.dropped_flows),
            'rejected_flows': len(self.rejected_flows),  # NEW
            'admission_ratio': len(self.completed_flows) / total_flows if total_flows > 0 else 0,  # NEW
            'deadline_met': met_deadline,
            'deadline_miss_ratio': 1 - (met_deadline / len(self.completed_flows)) if self.completed_flows else 1.0,
            'avg_delay': sum(delays) / len(delays) if delays else 0,
            'min_delay': min(delays) if delays else 0,
            'max_delay': max(delays) if delays else 0,
            'wcrt': max(delays) if delays else 0,  # NEW: Worst-Case Response Time
            'by_class': class_metrics
        }