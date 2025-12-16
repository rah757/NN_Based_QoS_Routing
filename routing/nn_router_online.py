# routing/nn_router_online.py - NN with online learning

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from ml.model import PathQualityNetwork
from ml.features import PathFeatureExtractor

class OnlineNNRouter:
    """NN router with online learning from experience"""
    
    def __init__(self, pretrained_model=None, epsilon=0.2, learning_rate=0.001):
        self.model = PathQualityNetwork(input_size=11)
        
        if pretrained_model:
            self.model.load_state_dict(torch.load(pretrained_model))
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = 0.995  # Decay exploration over time
        self.epsilon_min = 0.05
        
        self.feature_extractor = None
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=1000)
        self.batch_size = 32
        self.update_frequency = 10  # Update every N flows
        self.flow_count = 0
        
    def find_path(self, flow, G):
        """Find path with epsilon-greedy + experience collection"""
        
        if self.feature_extractor is None:
            self.feature_extractor = PathFeatureExtractor(G)
        
        # Get candidate paths
        candidate_paths = self.feature_extractor.get_candidate_paths(
            flow.source, flow.destination, k=3
        )
        
        if not candidate_paths:
            return None
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            chosen_path = random.choice(candidate_paths)
        else:
            # Use NN to select best path
            path_scores = []
            for path in candidate_paths:
                features = self.feature_extractor.extract_features(flow, path)
                if features is not None:
                    with torch.no_grad():
                        features_tensor = torch.from_numpy(features).unsqueeze(0)
                        score = self.model(features_tensor).item()
                        path_scores.append((path, score, features))
            
            if not path_scores:
                chosen_path = candidate_paths[0]
            else:
                # Select highest scoring path
                chosen_path = max(path_scores, key=lambda x: x[1])[0]
        
        return chosen_path
    
    def learn_from_experience(self, flow, path, G):
        """Update model based on flow outcome"""
        
        if not path or flow.completion_time is None:
            return
        
        # Extract features
        features = self.feature_extractor.extract_features(flow, path)
        if features is None:
            return
        
        # Label: did flow meet deadline?
        label = 1.0 if flow.deadline_met else 0.0
        
        # Add to replay buffer
        self.replay_buffer.append((features, label))
        
        self.flow_count += 1
        
        # Periodic training
        if self.flow_count % self.update_frequency == 0 and len(self.replay_buffer) >= self.batch_size:
            self._train_on_batch()
            
            # Decay exploration
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _train_on_batch(self):
        """Train on a random batch from replay buffer"""
        
        # Sample random batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        X_batch = np.array([x[0] for x in batch])
        y_batch = np.array([x[1] for x in batch])
        
        X_tensor = torch.from_numpy(X_batch)
        y_tensor = torch.from_numpy(y_batch).float().unsqueeze(1)
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(X_tensor)
        loss = self.criterion(outputs, y_tensor)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()