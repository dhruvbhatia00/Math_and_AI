import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

# --- Import our models ---
from betti_neural_network import BettiRegressorNN
from pytorch_mlp_model import BettiMLP, train_pytorch_mlp, evaluate_pytorch_mlp, predict_single_graph_mlp

# (Your helper functions: calculate_betti_number, generate_betti_data_..., etc. 
#  are all unchanged. They are omitted here for brevity, but 
#  YOU SHOULD KEEP THEM in your main.py file)
# ...
def calculate_betti_number(graph):
    """
    Calculates the first Betti number (cyclomatic number) of a graph.
    (This function is unchanged)
    """
    m = graph.number_of_edges()
    n = graph.number_of_nodes()
    c = nx.number_connected_components(graph)
    
    # print(f"Number of edges (m): {m}")
    # print(f"Number of vertices (n): {n}")
    # print(f"Number of connected components (c): {c}")
    
    return m - n + c

def generate_betti_data_uniform_edges(num_samples, num_vertices=10):
    """
    Generates random graphs with a uniform distribution of edge counts.
    """
    X, Y = [], []
    
    # Create a list of all possible edges for an undirected graph
    possible_edges = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            possible_edges.append((i, j))
    
    max_edges = len(possible_edges) # n*(n-1)/2
    
    for _ in range(num_samples):
        adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
        
        # 1. Choose a number of edges uniformly
        num_edges = random.randint(0, max_edges)
        
        # 2. Shuffle the list of possible edges and select the first 'num_edges'
        random.shuffle(possible_edges)
        selected_edges = possible_edges[:num_edges]
        
        # 3. Build the adjacency matrix
        for u, v in selected_edges:
            adj_matrix[u, v] = 1
            adj_matrix[v, u] = 1
            
        # 4. Calculate Betti number and store
        G = nx.from_numpy_array(adj_matrix)
        m, n, c = G.number_of_edges(), G.number_of_nodes(), nx.number_connected_components(G)
        betti_number = m - n + c
        
        X.append(adj_matrix.flatten())
        Y.append([betti_number])
        
    return np.array(X), np.array(Y)

def generate_betti_data_uniform_betti(num_samples, num_vertices=10, max_attempts=5_000_000):
    """
    Generates random graphs on `num_vertices` vertices, aiming for a uniform distribution
    over possible Betti numbers.
    (This function is unchanged)
    """
    X, Y = [], []

    # List all possible undirected edges
    possible_edges = [(i, j) for i in range(num_vertices) for j in range(i + 1, num_vertices)]
    max_edges = len(possible_edges)
    betti_buckets = {}
    target_per_betti = num_samples // 10  
    print(f"Target ≈ {target_per_betti} graphs per Betti number category")
    attempts = 0
    while len(X) < num_samples and attempts < max_attempts:
        attempts += 1
        num_edges = random.randint(0, max_edges)
        random.shuffle(possible_edges)
        selected_edges = possible_edges[:num_edges]
        adj_matrix = np.zeros((num_vertices, num_vertices), dtype=int)
        for u, v in selected_edges:
            adj_matrix[u, v] = 1
            adj_matrix[v, u] = 1
        G = nx.from_numpy_array(adj_matrix)
        betti = G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)
        if betti < 0: continue
        if betti not in betti_buckets:
            betti_buckets[betti] = 0
        if betti_buckets[betti] < target_per_betti:
            X.append(adj_matrix.flatten())
            Y.append([betti])
            betti_buckets[betti] += 1
        if len(X) >= num_samples:
            break
    print(f"Generated {len(X)} samples after {attempts} attempts.")
    print("Distribution of Betti numbers:")
    for k in sorted(betti_buckets):
        print(f"  β₁ = {k}: {betti_buckets[k]} samples")
    return np.array(X), np.array(Y)
# ...
# --- Define explicit graphs on 10 vertices (Unchanged) ---
def make_homeomorphic_graphs_on_10_vertices():
    graphs = {}

    # 1️⃣ Theta graph
    G_theta = nx.Graph()
    G_theta.add_edges_from([
        (0,1),(1,2),(2,9),
        (0,3),(3,4),(4,9),
        (0,5),(5,6),(6,7),(7,8),(8,9)
    ])
    graphs["Theta"] = G_theta

    # 2️⃣ Double theta graph
    G_double_theta = nx.Graph()
    G_double_theta.add_edges_from([
        (0,1),(1,2),(2,9),
        (0,3),(3,4),(4,9),
        (0,5),(5,6),(6,9),
        (0,7),(7,8),(8,9)
    ])
    graphs["Double Theta"] = G_double_theta

    # 3️⃣ Cube (8 vertices + 2 subdivisions)
    G_cube = nx.cubical_graph()
    while G_cube.number_of_nodes() < 10:
        u, v = list(G_cube.edges())[0]
        G_cube.remove_edge(u, v)
        new = max(G_cube.nodes()) + 1
        G_cube.add_node(new)
        G_cube.add_edges_from([(u, new), (new, v)])
    graphs["Cube"] = G_cube

    # 4️⃣ C10
    G_c10 = nx.cycle_graph(10)
    graphs["C10"] = G_c10

    # 5️⃣ C5 + Path
    G_c5_path = nx.cycle_graph(5)
    G_c5_path.add_nodes_from(range(5,10))
    G_c5_path.add_edges_from([(0,5),(5,6),(6,7),(7,8),(8,9)])
    graphs["C5+Path"] = G_c5_path

    # 6️⃣ Path Tree (a linear chain of 10)
    G_path = nx.path_graph(10)
    graphs["Path Tree"] = G_path

    # 7️⃣ Star Tree (center + 9 leaves)
    G_star = nx.star_graph(9)
    graphs["Star Tree"] = G_star

    # 8️⃣ Mercedes-Benz graph
    G_mb = nx.Graph()
    G_mb.add_edges_from([
        (0,4),(4,1),(1,5),(5,2),(2,6),(6,3),(3,7),(7,1),
        (0,8),(8,9),(9,2),(5,8)
    ])
    graphs["Mercedes-Benz"] = G_mb

    return graphs

# --- Predict + visualize for NumPy model (Unchanged) ---
def adjacency_matrix(G):
    """Return adjacency matrix as NumPy array with node ordering 0..n-1."""
    return nx.to_numpy_array(G, dtype=int)

def predict_and_visualize_numpy(graphs, model):
    print("\n" + "="*30)
    print("  RUNNING NUMPY (MLP) MODEL  ")
    print("="*30)
    for name, G in graphs.items():
        adj = adjacency_matrix(G)
        num_nodes = adj.shape[0]
        true_betti = G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)
        
        x = adj.flatten().reshape(1, num_nodes * num_nodes)
        predicted_betti = model.predict(x)[0][0]
        rounded_prediction = np.round(predicted_betti)

        print(f"\n===== {name} Graph =====")
        print(f"Nodes: {num_nodes}, Edges: {G.number_of_edges()}")
        print(f"True β₁ = {true_betti}")
        print(f"Predicted β₁ = {predicted_betti:.3f} (Rounded: {int(rounded_prediction)})")

        plt.figure(figsize=(6,6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color="#33a3ff",
                node_size=700, font_color="white", width=2.0)
        title_color = "green" if true_betti == rounded_prediction else "red"
        plt.title(f"{name} (NumPy Model)\nTrue β₁: {true_betti} | Pred: {int(rounded_prediction)}",
                  fontsize=14, color=title_color)
        plt.show()

        
# --- NEW: Predict + visualize for PyTorch MLP ---
def predict_and_visualize_mlp(graphs, model, device='cpu'):
    print("\n" + "="*30)
    print("  RUNNING PYTORCH (MLP) MODEL  ")
    print("="*30)
    for name, G in graphs.items():
        adj = adjacency_matrix(G)
        num_nodes = adj.shape[0]
        true_betti = G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)

        # Use our new single-graph prediction function
        x = adj.flatten().reshape(1, num_nodes * num_nodes)
        predicted_betti = predict_single_graph_mlp(model, x, device=device)
        rounded_prediction = np.round(predicted_betti)

        print(f"\n===== {name} Graph =====")
        print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        print(f"True β₁ = {true_betti}")
        print(f"Predicted β₁ = {predicted_betti:.3f} (Rounded: {int(rounded_prediction)})")

        plt.figure(figsize=(6,6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color="#ff6b6b", # Red color for this model
                node_size=700, font_color="white", width=2.0)
        title_color = "green" if true_betti == rounded_prediction else "red"
        plt.title(f"{name} (PyTorch MLP)\nTrue β₁: {true_betti} | Pred: {int(rounded_prediction)}",
                  fontsize=14, color=title_color)
        plt.show()


# --- Run everything ---
if __name__ == '__main__':

    # --- Hyperparameters ---
    NUM_VERTICES = 10
    INPUT_SIZE = NUM_VERTICES * NUM_VERTICES
    HIDDEN_SIZE = 128
    EPOCHS = 500 # Kept your 500 epochs

    # --- Check for GPU ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device.upper()}")

    # --- Generate Data (as NumPy arrays) ---
    print("Generating Betti number dataset...")
    X, Y = generate_betti_data_uniform_edges(
        num_samples=20000,
        num_vertices=NUM_VERTICES
    )

    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]
    print(f"Dataset created with {len(X_train)} training samples.")

    # --- 1. Train NumPy Network ---
    betti_nn = BettiRegressorNN(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE
    )
    print("\nStarting training (NumPy Model)...")
    betti_nn.train(X_train, Y_train, epochs=EPOCHS) # Using 500 epochs
    print("NumPy training finished.\n")


    # --- 2. Train PyTorch MLP Network ---
    pytorch_mlp = BettiMLP(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
    train_pytorch_mlp(pytorch_mlp, X_train, Y_train, epochs=EPOCHS, device=device)

    # --- Evaluate NumPy Model ---
    print("\nEvaluating NumPy model on the test set...")
    test_predictions_numpy = betti_nn.predict(X_test)
    mae_numpy = np.mean(np.abs(test_predictions_numpy - Y_test))
    print(f"NumPy Model MAE: {mae_numpy:.4f}")
    rounded_predictions_numpy = np.round(test_predictions_numpy)
    accuracy_numpy = np.mean(rounded_predictions_numpy == Y_test)
    print(f"NumPy Model Accuracy: {accuracy_numpy * 100:.2f}%")
    
    # --- Evaluate PyTorch MLP Model ---
    print("\nEvaluating PyTorch MLP model on the test set...")
    evaluate_pytorch_mlp(pytorch_mlp, X_test, Y_test, device=device)
    
    # --- Run Final Visualization Harness for BOTH models ---
    graphs = make_homeomorphic_graphs_on_10_vertices()
    
    # Model 1: NumPy
    predict_and_visualize_numpy(graphs, betti_nn)
    
    # Model 2: PyTorch MLP
    predict_and_visualize_mlp(graphs, pytorch_mlp, device=device)