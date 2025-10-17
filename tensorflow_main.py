import tensorflow as tf
import numpy as np
from main import generate_betti_data_uniform_betti

# TensorFlow model creation function
def create_betti_regressor(input_size, hidden_size=64):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_size,)),
        tf.keras.layers.Dense(hidden_size, activation='relu'),
        tf.keras.layers.Dense(hidden_size, activation='relu'),
        tf.keras.layers.Dense(1)  # Linear output for regression
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model


if __name__ == "__main__":
    # Hyperparameters
    NUM_VERTICES = 10
    INPUT_SIZE = NUM_VERTICES * NUM_VERTICES
    HIDDEN_SIZE = 64
    EPOCHS = 200
    NUM_SAMPLES = 20000

    # Generate dataset using your function
    print("Generating Betti number dataset...")
    X, Y = generate_betti_data_uniform_betti(NUM_SAMPLES, NUM_VERTICES)

    # Train/test split
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]

    print(f"Dataset created with {len(X_train)} training samples.")

    # Create TensorFlow model
    model = create_betti_regressor(INPUT_SIZE, HIDDEN_SIZE)

    # Train model
    print("\nStarting training...")
    model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=64, validation_split=0.2, verbose=2)
    print("Training finished.\n")

    # Evaluate on test set
    print("Evaluating model on the test set...")
    loss, mae = model.evaluate(X_test, Y_test, verbose=2)
    print(f"Mean Absolute Error (MAE) on test set: {mae:.4f}")

    # Example prediction on test set
    predictions = model.predict(X_test)
    rounded_predictions = np.round(predictions)
    accuracy = np.mean(rounded_predictions == Y_test)
    print(f"Accuracy of rounded predictions: {accuracy * 100:.2f}%")
