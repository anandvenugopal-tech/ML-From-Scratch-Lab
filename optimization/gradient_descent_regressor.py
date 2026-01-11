"""
Gradient Descent from Scratch

This module implements the gradient descent algorithm to minimize Mean Squared Error (MSE)
loss for a simple linear regression model.

"""

#import numpy library
import numpy as np

#define a class gradient descent.
class GradientDescent:
    def __init__(self, learning_rate, n_iter, tolerance):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.tolerance = tolerance
        self.loss_history = []

    #define loss function
    def _mse_loss(self, y, y_pred):
        return np.mean((y_pred - y) ** 2)

    #define fit function for updating parameters
    def fit(self, x, y): 
        n_samples, n_features = x.shape

        #initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        prev_loss = float("inf")

        for _ in range(self.n_iter):

            # predictions
            y_pred = np.dot(x, self.weights) + self.bias

            # compute loss
            loss = self._mse_loss(y, y_pred)
            self.loss_history.append(loss)

            if abs(prev_loss - loss) < self.tolerance:
                break

            prev_loss = loss

            # compute gradients
            dw = (2 / n_samples) * np.dot(x.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)

            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db



