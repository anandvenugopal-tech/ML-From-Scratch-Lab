"""
Logistic Regression From Scratch
--------------------------------

This module implements binary Logistic Regression using
Gradient Descent for optimization.

No machine learning libraries are used.
Only NumPy + your own gradient descent implementation.

Logistic regression is a classification algorithm, not a regression algorithm.
It predicts the probability of class 1 using the sigmoid function.
"""

import numpy as np
from optimization.gradient_descent import GradientDescent

#define sigmoid function
def sigmoid(z):
    # sigmoid function is probabilistic function that map the real values into the range (0, 1).
    # σ(z) = 1 / (1 + e^-z)
    return 1 / (1 + np.exp(-z))

#define class Logistic Regression
class LogisticRegression:

    #initialize parameter
    def __init__(self, learning_rate = 0.01, n_iter = 1000, tolerance = 1e-6):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.tolerance = tolerance

        #create optimizer
        self.optimizer = GradientDescent(
            learning_rate = self.learning_rate,
            n_iter = n_iter,
            tolerance= self.tolerance
        )

        self.weights = None
        self.bias = None

    # Compute Binary Cross-Entropy (Log Loss).
    def binary_cross_entropy(self, y, y_pred):
        # L = -1/n Σ [y log(y_pred) + (1-y) log(1 - y_pred)]
        #clip predictions to avoid log(0)
        eps = 1e-10
        y_pred = np.clip(y_pred, eps, 1-eps)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    #define fit function to train the model
    def fit(self, x, y):
        n_samples, n_features = x.shape

        #initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        prev_loss = float('inf')

        for _ in range(self.n_iter):

            #linear combination
            z = np.dot(x, self.weights) + self.bias

            #apply sigmoid to get probabilities
            y_pred = sigmoid(z)

            #compute loss
            loss = self.binary_cross_entropy(y, y_pred)

            if abs(prev_loss - loss) < self.tolerance:
                break

            prev_loss = loss

            #compute gradients
            dw = (1/n_samples) * np.dot(x.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            #update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def probability(self, x):
        return sigmoid(np.dot(x, self.weights) + self.bias)

    def predict(self,x):
        probabilities = self.probability(x)
        return np.where(probabilities > 0.5, 1, 0)


if __name__ == "__main__":
    # Simple OR logic dataset
    x = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([0, 1, 1, 1])  # OR output

    model = LogisticRegression(learning_rate=0.1, n_iter=5000)
    model.fit(x, y)

    print("Weights:", model.weights)
    print("Bias:", model.bias)

    print("Predictions:", model.predict(x))
    print("Probabilities:", model.probability(x))