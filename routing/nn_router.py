# routing/nn_router.py - NN-based router

import torch
import numpy as np
import random
from ml.model import PathQualityNetwork
from ml.features import PathFeatureExtractor

class NNRouter:
    """Neural network-based QoS router"""
    
    def __init__(self, pretrained_model=None, epsilon=0.1):
        """
        Args:
            pretrained_model: Path to pretrained model weights
            epsilon: Exploration rate (0.1 = 10% random exploration)
        """
        self.model = PathQualityNetwork(input_size=13)
        
        if pretrained_model:
            self.model.load_state_dict(torch.load(pretrained_model))
        
        self.model.eval()
        self.epsilon = epsilon
        self.feature_extractor = None
        
    def find_path(self, flow, G):
        """Find best path using neural network"""
        
        # Initialize feature extractor if needed
        if self.feature_extractor is None:
            self.feature_extractor = PathFeatureExtractor(G)
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Random path selection
            candidate_paths = self.feature_extractor.get_candidate_paths(
                flow.source, flow.destination, k=3
            )
            if not candidate_paths:
                return None
            return random.choice(candidate_paths)
        
        # Get candidate paths
        candidate_paths = self.feature_extractor.get_candidate_paths(
            flow.source, flow.destination, k=3
        )
        
        if not candidate_paths:
            return None
        
        # Extract features for each path
        path_scores = []
        for path in candidate_paths:
            features = self.feature_extractor.extract_features(flow, path)
            if features is not None:
                # Predict quality score
                with torch.no_grad():
                    features_tensor = torch.from_numpy(features).unsqueeze(0)
                    score = self.model(features_tensor).item()
                    path_scores.append((path, score))
        
        if not path_scores:
            return candidate_paths[0]  # fallback to first path
        
        # Select path with highest predicted quality
        best_path = max(path_scores, key=lambda x: x[1])[0]
        return best_path