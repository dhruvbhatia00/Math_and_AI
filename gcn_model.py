import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx

# --- PyTorch Geometric Imports ---
# Make sure you have run 'pip install torch-geometric'
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

class BettiGCN(nn.Module):
    """
    A Graph Convolutional Network (GCN) to predict Betti numbers.
    It operates directly on graph-structured data.
    """
    def __init__(self, in_channels=10, hidden_channels=64): # <-- FIX 1: WAS 1
        super(BettiGCN, self).__init__()
        # Node features have 10 dimensions (one-hot identity vector)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.conv3 = GCNConv(hidden_channels * 2, hidden_channels * 2)

        # Fully connected layers for graph-level regression
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        # Unpack the data object
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. GCN layers
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)

        # 2. Global Pooling
        # Aggregates node features into a single graph-level feature vector
        x = global_mean_pool(x, batch)
        
        # 3. Fully Connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x) # Final regression output
        
        return x

def create_pyg_dataset(X_flat_matrices, Y_labels, num_vertices=10):
    """
    Converts the NumPy dataset (adjacency matrices) into a list
    of PyTorch Geometric `Data` objects.
    """
    pyg_dataset = []
    
    # Create Identity matrix for node features
    I = torch.eye(num_vertices, dtype=torch.float32)
    
    for i in range(len(X_flat_matrices)):
        adj_matrix = X_flat_matrices[i].reshape(num_vertices, num_vertices)
        
        # --- FIX 2: Fixed UserWarning for performance ---
        y = torch.tensor(Y_labels[i], dtype=torch.float32) # WAS: [Y_labels[i]]
        
        # Use Identity matrix for features
        x_nodes = I
        
        # Convert adjacency matrix to edge_index format
        rows, cols = np.nonzero(adj_matrix)
        edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
        
        data = Data(x=x_nodes, edge_index=edge_index, y=y)
        pyg_dataset.append(data)
        
    return pyg_dataset

def train_gcn_model(model, train_dataset, epochs=200, batch_size=64, device='cpu'):
    """
    Trains the PyTorch Geometric GCN model.
    """
    model.to(device)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    print("Starting PyTorch GCN training (with Identity Features)...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs} - Training MSE Loss: {avg_loss:.4f}")
            
    print("GCN training finished.\n")

def evaluate_gcn_model(model, test_dataset, device='cpu'):
    """
    Evaluates the trained GCN model on the test set.
    """
    model.to(device)
    model.eval()
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            predictions.append(outputs.cpu().numpy())
            true_labels.append(batch.y.cpu().numpy())
            
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - true_labels))
    rounded_predictions = np.round(predictions)
    accuracy = np.mean(rounded_predictions == true_labels)
    
    print("--- PyTorch GCN Model Evaluation ---")
    print(f"Mean Absolute Error (MAE) on test set: {mae:.4f}")
    print(f"Accuracy of rounded predictions: {accuracy * 100:.2f}%")

def predict_single_graph_gcn(model, G: nx.Graph, device='cpu'):
    """
    Predicts the Betti number for a single NetworkX graph.
    """
    model.to(device)
    model.eval()
    
    num_vertices = G.number_of_nodes()
    
    # Use Identity matrix for node features
    x_nodes = torch.eye(num_vertices, dtype=torch.float32)
    
    # Convert NetworkX graph to edge_index
    adj_matrix = nx.to_numpy_array(G, dtype=int)
    rows, cols = np.nonzero(adj_matrix)
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    
    # Create a Data object and send to device
    data = Data(x=x_nodes, edge_index=edge_index).to(device)
    
    # 2. Predict
    with torch.no_grad():
        prediction = model(data)
        
    return prediction.cpu().item()