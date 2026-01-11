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
from optimization.gradient_descent_classifier import GradientDescent

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

    def fit(self, x, y):
        self.optimizer.fit(x, y)
        self.weights = self.optimizer.weights
        self.bias = self.optimizer.bias

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