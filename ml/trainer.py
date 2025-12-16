# ml/trainer.py - Train NN from baseline data
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ml.model import PathQualityNetwork
from ml.features import PathFeatureExtractor
import simpy
from network.topology import WANTopology
from traffic.generator import TrafficGenerator
from routing.dijkstra import DijkstraRouter
from simulation.simulator import NetworkSimulator

class NNTrainer:
    """Train neural network from collected routing data"""
    
    def __init__(self):
        self.model = PathQualityNetwork(input_size=11)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        
    def collect_training_data(self, num_samples=500):
        """Collect training data by running baseline simulation"""
        print("Collecting training data...")
        
        env = simpy.Environment()
        topology = WANTopology(n_nodes=20, seed=42)
        router = DijkstraRouter()
        traffic_gen = TrafficGenerator(env, topology.get_graph(), 
                                      arrival_rate=10, load_factor=0.6)
        
        sim = NetworkSimulator(env, topology, traffic_gen, router)
        sim.run()
        env.run(until=50)  # Run longer to collect more data
        
        # Extract features and labels from completed flows
        feature_extractor = PathFeatureExtractor(topology.get_graph())
        
        X_train = []
        y_train = []
        
        for flow in sim.completed_flows:
            if flow.path_taken:
                features = feature_extractor.extract_features(flow, flow.path_taken)
                if features is not None:
                    X_train.append(features)
                    # Label: 1 if deadline met, 0 if missed
                    y_train.append(1.0 if flow.deadline_met else 0.0)
        
        print(f"Collected {len(X_train)} training samples")
        print(f"  Positive (deadline met): {sum(y_train)}")
        print(f"  Negative (deadline missed): {len(y_train) - sum(y_train)}")
        
        return np.array(X_train), np.array(y_train)
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """Train the model"""
        print(f"\nTraining for {epochs} epochs...")
        
        X_tensor = torch.from_numpy(X_train)
        y_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
        
    def save_model(self, path='models/nn_router.pth'):
        """Save trained model"""
        import os
        os.makedirs('models', exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"\nModel saved to {path}")

if __name__ == "__main__":
    trainer = NNTrainer()
    X_train, y_train = trainer.collect_training_data(num_samples=500)
    trainer.train(X_train, y_train, epochs=50)
    trainer.save_model()