import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class BettiMLP(nn.Module):
    """
    A PyTorch-based 'vanilla' neural network (MLP) with 3 hidden layers.
    It takes a flattened adjacency matrix as input.
    """
    def __init__(self, input_size=100, hidden_size=128, output_size=1):
        super(BettiMLP, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # nn.Linear(hidden_size, hidden_size * 2), # A wider second layer
            # nn.ReLU(),
            # nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            
            nn.Linear(hidden_size*2, output_size) # Regression output
        )

    def forward(self, x):
        return self.layers(x)

def train_pytorch_mlp(model, X_train, Y_train, epochs=200, batch_size=64, device='cpu'):
    """
    Trains the PyTorch MLP model.
    """
    model.to(device)
    
    # --- Prepare Data ---
    # Data is already flat, so no reshaping needed
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    
    dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # --- Loss and Optimizer ---
    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting PyTorch MLP training...")
    for epoch in range(epochs):
        model.train() # Set model to training mode
        epoch_loss = 0.0
        
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs} - Training MSE Loss: {avg_loss:.4f}")
            
    print("PyTorch MLP training finished.\n")

def evaluate_pytorch_mlp(model, X_test, Y_test, device='cpu'):
    """
    Evaluates the trained PyTorch MLP model on the test set.
    """
    model.to(device)
    model.eval() # Set model to evaluation mode
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32) # Keep on CPU for numpy
    
    with torch.no_grad(): # Disable gradient calculations
        test_predictions_tensor = model(X_test_tensor)
    
    test_predictions = test_predictions_tensor.cpu().numpy()
    
    # Calculate metrics
    mae = np.mean(np.abs(test_predictions - Y_test))
    rounded_predictions = np.round(test_predictions)
    accuracy = np.mean(rounded_predictions == Y_test)
    
    print("--- PyTorch MLP Model Evaluation ---")
    print(f"Mean Absolute Error (MAE) on test set: {mae:.4f}")
    print(f"Accuracy of rounded predictions: {accuracy * 100:.2f}%")

def predict_single_graph_mlp(model, flat_adj_matrix, device='cpu'):
    """
    Predicts the Betti number for a single flattened adjacency matrix.
    """
    model.to(device)
    model.eval()
    
    adj_tensor = torch.tensor(flat_adj_matrix, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        prediction = model(adj_tensor)
        
    return prediction.cpu().item() # Return as a single Python number