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

def visualize_betti_prediction(adj_matrix_vector, true_betti, predicted_betti):
    """
    Visualizes a graph, its true Betti number, and the model's prediction.
    """
    num_vertices = int(np.sqrt(len(adj_matrix_vector)))
    adj_matrix = adj_matrix_vector.reshape((num_vertices, num_vertices))
    
    G = nx.from_numpy_array(adj_matrix)
    
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='#33a3ff', node_size=500, font_color='white', width=1.5)
    
    rounded_prediction = np.round(predicted_betti)
    is_correct = (true_betti == rounded_prediction)
    
    title_color = 'green' if is_correct else 'red'
    title = (f"True Betti Number: {true_betti}\n"
             f"Model Prediction: {predicted_betti:.2f} (Rounded: {int(rounded_prediction)})")
    
    plt.title(title, color=title_color, fontsize=14)
    # The plt.show() command will be called outside the function
    # to display the plots one by one.

if __name__ == '__main__':

    # --- Hyperparameters ---
    NUM_VERTICES = 10
    INPUT_SIZE = NUM_VERTICES * NUM_VERTICES
    HIDDEN_SIZE = 64
    EPOCHS = 200
    
    # --- Generate Data ---
    print("Generating Betti number dataset...")
    X, Y = generate_betti_data_uniform_edges(num_samples=20000, num_vertices=NUM_VERTICES)
    
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]

    print(f"Dataset created with {len(X_train)} training samples.")

    # --- 1. Train NumPy Network ---
    betti_nn = BettiRegressorNN(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE
    )
    
    print("\nStarting training...")
    betti_nn.train(X_train, Y_train, epochs=EPOCHS)
    print("Training finished.\n")
    
    # --- Evaluate the Model ---
    print("Evaluating model on the test set...")
    test_predictions = betti_nn.predict(X_test)
    
    # Calculate Mean Absolute Error (how far off is the prediction on average)
    mae = np.mean(np.abs(test_predictions - Y_test))
    print(f"Mean Absolute Error (MAE) on test set: {mae:.4f}")

    # For a more intuitive measure, check accuracy of rounded predictions
    rounded_predictions = np.round(test_predictions)
    accuracy = np.mean(rounded_predictions == Y_test)
    print(f"Accuracy of rounded predictions: {accuracy * 100:.2f}%")
    
    # --- Show some examples ---
    # print("\n--- Example Predictions (Predicted vs. True) ---")
    # for i in range(5):
    #     idx = np.random.randint(0, len(X_test))
    #     pred = test_predictions[idx][0]
    #     true = Y_test[idx][0]
    #     print(f"Prediction: {pred:.2f} (Rounded: {np.round(pred)}) | True Value: {true}")

    # print("\n--- Visualizing a few test examples ---")

    # # Let's check 4 random examples from the test set
    # for _ in range(4):
    #     # Select a random index from the test set
    #     idx = np.random.randint(0, len(X_test))
    #     graph_vector = X_test[idx]
    #     true_label = Y_test[idx][0]
    #     prediction = test_predictions[idx][0]
        
    #     print(f"\nVisualizing Test Example - True Betti: {true_label}, Predicted: {prediction:.2f}")
    #     # Visualize the result
    #     visualize_betti_prediction(graph_vector, true_label, prediction)
    #     plt.show() # Display the plot

    betti_nn = BettiRegressorNN(input_size=INPUT_SIZE, hidden_size=128)
    betti_nn.train(X_train, Y_train, epochs=500)

    # --- 2. Generate the Adjacency Matrix for the new graph ---
    adj_matrix = np.zeros((NUM_VERTICES, NUM_VERTICES), dtype=int)
    # Create C_10 cycle edges
    for i in range(NUM_VERTICES):
        adj_matrix[i, (i + 1) % NUM_VERTICES] = 1
    
    # Connect ONLY one pair of opposite vertices (0 and 5)
    adj_matrix[0, 5] = 1
    adj_matrix[1,4] =1
    
    # Make the matrix symmetric
    adj_matrix = adj_matrix + adj_matrix.T

    # --- 3. Calculate the True Betti Number ---
    G = nx.from_numpy_array(adj_matrix)
    true_betti = G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)

    print("--- Graph Properties ---")
    print(f"True First Betti Number (β1): {true_betti}\n")

    # --- 4. Have the Neural Network Evaluate it ---
    graph_vector = adj_matrix.flatten().reshape(1, -1)
    predicted_betti = betti_nn.predict(graph_vector)[0][0]
    rounded_prediction = np.round(predicted_betti)

    print("--- Neural Network Evaluation ---")
    print(f"Model Prediction: {predicted_betti:.4f}")
    print(f"Rounded Prediction: {int(rounded_prediction)}")
    if rounded_prediction == true_betti:
        print("\nThe model correctly predicted the Betti number! ✅")
    else:
        print("\nThe model's prediction was incorrect. ❌")

    # --- 5. Visualize the graph ---
    plt.figure(figsize=(7, 7))
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='#33a3ff', node_size=700, font_color='white', width=2.0)
    title = (f"True $\\beta_1$: {true_betti} | Predicted $\\beta_1$: {int(rounded_prediction)}")
    plt.title(title, fontsize=16)
    plt.show()

    