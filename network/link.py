# network/link.py - Add dynamic link state

class Link:
    """Represents a network link with dynamic state"""
    
    def __init__(self, bandwidth, delay, jitter, loss):
        self.max_bandwidth = bandwidth  # Mbps
        self.base_delay = delay         # ms
        self.jitter = jitter            # ms
        self.loss_rate = loss           # %
        
        # Dynamic state
        self.current_flows = []
        self.queue = []
        self.queue_delay = 0  # ms
        
    def get_available_bandwidth(self):
        """Calculate currently available bandwidth"""
        used_bw = sum(f.bandwidth_mbps for f in self.current_flows)
        return max(0, self.max_bandwidth - used_bw)
    
    def get_current_delay(self):
        """Get current delay including queuing"""
        # Queue delay increases with utilization
        utilization = 1 - (self.get_available_bandwidth() / self.max_bandwidth)
        queue_delay = self.base_delay * (utilization ** 2) * 5  # nonlinear increase
        return self.base_delay + queue_delay
    
    def add_flow(self, flow):
        """Add active flow to link"""
        self.current_flows.append(flow)
    
    def remove_flow(self, flow):
        """Remove completed flow"""
        if flow in self.current_flows:
            self.current_flows.remove(flow)