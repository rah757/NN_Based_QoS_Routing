# ml/model.py - Neural Network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

class PathQualityNetwork(nn.Module):
    """Neural network to predict path quality (probability of meeting deadline)"""
    
    def __init__(self, input_size=11):
        super(PathQualityNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # Output: probability 0-1
        return x