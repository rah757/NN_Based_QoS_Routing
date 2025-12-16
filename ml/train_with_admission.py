# ml/train_with_admission.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import simpy
from network.topology import WANTopology
from traffic.generator import TrafficGenerator
from routing.min_delay import MinDelayRouter
from simulation.simulator import NetworkSimulator  # Now has admission control!
from ml.model import PathQualityNetwork
from ml.features import PathFeatureExtractor

def collect_realtime_training_data():
    """Collect training data WITH admission control enabled"""
    
    print("="*80)
    print("COLLECTING REAL-TIME TRAINING DATA (with Admission Control)")
    print("="*80)
    
    all_X = []
    all_y = []
    
    # Collect from multiple load levels
    for load in [0.3, 0.5, 0.7, 0.9]:
        print(f"\nCollecting from {load*100:.0f}% load...")
        
        env = simpy.Environment()
        topology = WANTopology(n_nodes=20, seed=42)
        router = MinDelayRouter()
        traffic_gen = TrafficGenerator(env, topology.get_graph(), 
                                      arrival_rate=15, load_factor=load)
        
        # THIS NOW USES ADMISSION CONTROL
        sim = NetworkSimulator(env, topology, traffic_gen, router)
        sim.run()
        env.run(until=100)
        
        feature_extractor = PathFeatureExtractor(topology.get_graph())
        
        # Extract features from COMPLETED flows (passed admission)
        for flow in sim.completed_flows:
            if flow.path_taken:
                features = feature_extractor.extract_features(flow, flow.path_taken)
                if features is not None:
                    all_X.append(features)
                    all_y.append(1.0 if flow.deadline_met else 0.0)
        
        # Also learn from REJECTED flows (negative examples)
        for flow in sim.rejected_flows:
            # Try to extract features for the path that was rejected
            try:
                path = router.find_path(flow, topology.get_graph())
                if path:
                    features = feature_extractor.extract_features(flow, path)
                    if features is not None:
                        all_X.append(features)
                        all_y.append(0.0)  # Label as "bad" since it was rejected
            except:
                pass
        
        print(f"  Completed: {len(sim.completed_flows)}")
        print(f"  Rejected: {len(sim.rejected_flows)}")
    
    X = np.array(all_X)
    y = np.array(all_y)
    
    print(f"\n{'='*80}")
    print(f"Total training samples: {len(X)}")
    print(f"Positive samples: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"Negative samples: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    print(f"{'='*80}")
    
    return X, y

def train_realtime_model(X, y, epochs=150):
    """Train with focus on real-time metrics"""
    
    print("\nTraining real-time aware model...")
    
    model = PathQualityNetwork(input_size=13)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    # Use weighted loss - penalize false positives more
    # (admitting unschedulable flows is worse than rejecting schedulable ones)
    criterion = nn.BCELoss()
    
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val).float().unsqueeze(1)
    
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    best_val_loss = float('inf')
    patience = 20
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            predictions = (val_outputs > 0.5).float()
            accuracy = (predictions == y_val_t).float().mean().item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss/len(dataloader):.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {accuracy*100:.1f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'models/nn_router_realtime.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print(f"\nBest model saved to 'models/nn_router_realtime.pth'")
    return model

if __name__ == "__main__":
    X, y = collect_realtime_training_data()
    model = train_realtime_model(X, y, epochs=150)