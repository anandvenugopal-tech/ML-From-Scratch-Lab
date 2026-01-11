"""
Linear Regression From Scratch

This module implements Linear Regression using the GradientDescent optimizer.
No machine learning libraries are used.
"""

#import required libraries
import numpy as np
from optimization.gradient_descent_regressor import GradientDescent

#define class Linear Regression
class LinearRegression:
	#initialize parameters
	def __init__(self,learning_rate = 0.01, n_iter = 1000, tolerance = 1e-6):
		self.learning_rate = learning_rate
		self.n_iter = n_iter
		self.tolerance = tolerance

		#create the optimizer
		self.optimizer = GradientDescent(
			learning_rate = self.learning_rate,
			n_iter=self.n_iter,
			tolerance=self.tolerance
		)
		self.weights = None
		self.bias = None

	#define fit function for train the model
	def fit(self, x, y):
		self.optimizer.fit(x, y)
		self.weights = self.optimizer.weights
		self.bias = self.optimizer.bias

	#define predict function for make predictions
	def predict(self, x):
		return np.dot(x, self.weights) + self.bias


#testing the implementation
if __name__ == '__main__':

	#create a simple dataset
	x = np.array([[1], [2], [3], [4], [5]])
	y = np.array([3, 5, 7, 9, 11])

	#create and train the model
	model = LinearRegression()
	model.fit(x, y)

	#print everything
	print(f'Weights: {model.weights}')
	print(f'Bias: {model.bias}')
	print(f'Predictions: {model.predict(x)}')



