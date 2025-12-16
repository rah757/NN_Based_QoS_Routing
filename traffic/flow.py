# traffic/flow.py

import time
from enum import Enum

class TrafficClass(Enum):
    VIDEO = "video"
    VOIP = "voip"
    INDUSTRIAL = "industrial"

class Flow:
    """Represents a single data flow with real-time constraints"""
    
    def __init__(self, flow_id, source, destination, traffic_class, 
                 size_bytes, creation_time):
        self.flow_id = flow_id
        self.source = source
        self.destination = destination
        self.traffic_class = traffic_class
        self.size_bytes = size_bytes
        self.creation_time = creation_time
        
        # Set deadline and bandwidth based on traffic class
        self._set_qos_parameters()
        
        # Track flow state
        self.completion_time = None
        self.path_taken = []
        self.deadline_met = None
        
    def _set_qos_parameters(self):
        """Set QoS requirements based on traffic class"""
        import random
        
        if self.traffic_class == TrafficClass.VIDEO:
            self.deadline_ms = 150  # 150ms max delay
            self.max_jitter_ms = 30
            self.bandwidth_mbps = random.uniform(2, 8)
            self.priority = 1  # lowest
            
        elif self.traffic_class == TrafficClass.VOIP:
            self.deadline_ms = 50   # 50ms max delay
            self.max_jitter_ms = 10
            self.bandwidth_mbps = random.uniform(0.064, 0.128)
            self.priority = 2  # medium
            
        elif self.traffic_class == TrafficClass.INDUSTRIAL:
            self.deadline_ms = 10   # 10ms ultra-low latency
            self.max_jitter_ms = 2
            self.bandwidth_mbps = 0.1
            self.priority = 3  # highest
    
    def get_absolute_deadline(self):
        """Return absolute deadline in simulation time"""
        return self.creation_time + (self.deadline_ms / 1000.0)
    
    def check_deadline(self, current_time):
        """Check if flow met its deadline"""
        if self.completion_time is None:
            return None
        
        delay = (self.completion_time - self.creation_time) * 1000  # convert to ms
        self.deadline_met = delay <= self.deadline_ms
        return self.deadline_met
    
    def get_delay_ms(self):
        """Get end-to-end delay in milliseconds"""
        if self.completion_time is None:
            return None
        return (self.completion_time - self.creation_time) * 1000
    
    def __repr__(self):
        return (f"Flow(id={self.flow_id}, {self.source}→{self.destination}, "
                f"{self.traffic_class.value}, deadline={self.deadline_ms}ms)")