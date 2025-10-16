import numpy as np

class BettiRegressorNN:
    def __init__(self, input_size, hidden_size, output_size=1, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Standard weight initialization
        self.W1 = np.random.randn(self.hidden_size, self.input_size) * np.sqrt(2. / self.input_size)
        self.b1 = np.zeros((self.hidden_size, 1))
        self.W2 = np.random.randn(self.output_size, self.hidden_size) * np.sqrt(1. / self.hidden_size)
        self.b2 = np.zeros((self.output_size, 1))

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        return Z > 0

    def _forward_pass(self, X):
        X = X.T
        Z1 = self.W1.dot(X) + self.b1
        A1 = self._relu(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = Z2  # Linear activation for regression output
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def _backward_pass(self, X, Y, cache):
        m = X.shape[0]
        X, Y = X.T, Y.T
        A1, A2 = cache['A1'], cache['A2']
        Z1 = cache['Z1']

        # Derivative for Mean Squared Error loss
        dZ2 = A2 - Y
        dW2 = (1/m) * dZ2.dot(A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        
        dA1 = self.W2.T.dot(dZ2)
        dZ1 = dA1 * self._relu_derivative(Z1)
        dW1 = (1/m) * dZ1.dot(X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        
        gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return gradients

    def _update_parameters(self, gradients):
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']

    def train(self, X_train, Y_train, epochs=200, batch_size=64):
        for epoch in range(epochs):
            permutation = np.random.permutation(X_train.shape[0])
            X_shuffled, Y_shuffled = X_train[permutation], Y_train[permutation]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch, Y_batch = X_shuffled[i:i+batch_size], Y_shuffled[i:i+batch_size]
                A2, cache = self._forward_pass(X_batch)
                gradients = self._backward_pass(X_batch, Y_batch, cache)
                self._update_parameters(gradients)

            if (epoch + 1) % 20 == 0:
                predictions, _ = self._forward_pass(X_train)
                loss = np.mean((predictions - Y_train.T)**2) # Calculate MSE
                print(f"Epoch {epoch+1}/{epochs} - Training MSE Loss: {loss:.4f}")

    def predict(self, X):
        predictions, _ = self._forward_pass(X)
        return predictions.T