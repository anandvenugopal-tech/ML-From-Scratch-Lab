import numpy as np
from tqdm import tqdm


def sigmoid(z: np.ndarray) -> np.ndarray:
  return 1.0/(1.0 + np.exp(-z))
def relu(z: np.ndarray) -> np.ndarray:
  return np.maximum(0, z)
def tanh(z: np.ndarray) -> np.ndarray:
  return np.tanh(z)
def leaky_relu(z: np.ndarray) -> np.ndarray:
  return np.where(z > 0, z, z * 0.01)

def softmax(z: np.ndarray) -> np.ndarray:
  e = np.exp(z - np.max(z, axis=0, keepdims=True))
  return e / np.sum(e, axis = 0)

def normalize(x: np.ndarray) -> np.ndarray:
  return (x - np.min(x))/(np.max(x) - np.min(x))

def one_hot_encode(x: np.ndarray, num_labels: int) -> np.ndarray:
  return np.eye(num_labels)[x]

def derivative(function_name: str, z: np.ndarray) -> np.ndarray:
  if function_name == 'sigmoid':
    return sigmoid(z) * (1 - sigmoid(z))
  if function_name == 'tanh':
    return 1 - np.square(tanh(z))
  if function_name == 'relu':
    y = (z > 0) * 1
    return y
  if function_name == 'leaky_relu':
    return np.where(z > 0, 1, 0.01)
  return 'No such activation'

class NeuralNetwork:
  def __init__(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, activation: str, num_labels: int, architecture: list[int]):
    self.X = normalize(X) # normalize traning data in range 0,1
    assert np.all((self.X >= 0) | (self.X <= 1)) # test that normalize succeed
    self.X, self.X_test = X.copy(), X_test.copy()
    self.y, self.y_test = y.copy(), y_test.copy()
    self.layers = {} # define dict to store results of activation
    self.architecture = architecture # size of the hidden layer as array
    self.activation = activation # activation function
    assert self.activation in ['relu', 'tanh', 'sigmoid', 'leaky_relu']
    self.parameters = {}
    self.num_labels = num_labels
    self.m = X.shape[1]
    self.architecture.append(self.num_labels)
    self.num_input_features = X.shape[0]
    self.architecture.insert(0, self.num_input_features)
    self.L = len(architecture)
    assert self.X.shape == (self.num_input_features, self.m)
    assert self.y.shape == (self.num_labels, self.m)

  def initialize_parameters(self):
    for i in range(1, self.L):
      print(f'Initializing parameters for layer: {i}.')
      self.parameters['w'+ str(i)] = np.random.randn(self.architecture[i], self.architecture[i-1]) * 0.01
      self.parameters['b'+ str(i)] = np.zeros((self.architecture[i], 1))

  def forward(self):
    params = self.parameters
    self.layers['a0'] = self.X
    for l in range(1, self.L - 1):
      self.layers['z'+str(l)] = np.dot(params['w'+str(l)], self.layers['a'+str(l-1)]) + params['b' + str(l)]
      self.layers['a'+str(l)] = eval(self.activation)(self.layers['z' + str(l)])
      assert self.layers['a'+str(l)].shape == (self.architecture[l], self.m)
    self.layers['z' + str(self.L-1)] = np.dot(params['w' + str(self.L-1)], self.layers['a' + str(self.L-2)]) + params['b' + str(self.L-1)]
    self.layers['a' + str(self.L-1)] = softmax(self.layers['z' + str(self.L-1)])
    self.output = self.layers['a'+str(self.L-1)]
    assert self.output.shape == (self.num_labels, self.m)
    assert all([s for s in np.sum(self.output, axis = 1)])

    cost = - np.sum(self.y * np.log(self.output + 1e-9)) / self.m

    return cost, self.layers

  def backpropagate(self):
    derivatives = {}
    dZ = self.output - self.y
    assert dZ.shape == (self.num_labels, self.m)
    dW = np.dot(dZ, self.layers['a' + str(self.L-2)].T) / self.m
    db = np.sum(dZ, axis = 1, keepdims = True) / self.m
    dAPrev = np.dot(self.parameters['w' + str(self.L-1)].T, dZ)
    derivatives['dW' + str(self.L-1)] = dW
    derivatives['db' + str(self.L-1)] = db

    for l in range(self.L-2, 0, -1):
      dZ = dAPrev * derivative(self.activation, self.layers['z' + str(l)])
      dW = 1. / self.m * np.dot(dZ, self.layers['a' + str(l-1)].T)
      db = 1. / self.m * np.sum(dZ, axis = 1, keepdims = True)
      if l > 1:
        dAPrev = np.dot(self.parameters['w' + str(l)].T, (dZ))
      derivatives['dW' + str(l)] = dW
      derivatives['db' + str(l)] = db
    self.derivatives = derivatives

    return self.derivatives

  def fit(self, X, y, lr=0.01, epochs = 1000):
    self.costs = []
    self.initialize_parameters()
    self.accuaracies = {'train': [], 'test': []}
    for epoch in tqdm(range(epochs), colour = 'BLUE'):
      cost, cache = self.forward()
      self.costs.append(cost)
      derivatives = self.backpropagate()
      for layer in range(1, self.L):
        self.parameters['w' + str(layer)] = self.parameters['w' + str(layer)] - lr * self.derivatives['dW' + str(layer)]
        self.parameters['b' + str(layer)] = self.parameters['b' + str(layer)] - lr * self.derivatives['db' + str(layer)]
      train_accuracy = self.accuracy(self.X, self.y)
      test_accuracy = self.accuracy(self.X_test, self.y_test)
      if epoch % 10 == 0:
        print(f'Epoch: {epoch:3d} | Cost: {cost: .3f} | Accuracy: {train_accuracy: .3f}')
        self.accuaracies['train'].append(train_accuracy)
        self.accuaracies['test'].append(test_accuracy)
    print('Traing terminated')

  def predict(self, x):
    params = self.parameters
    n_layers = self.L - 1
    values = [x]
    for l in range(1, n_layers):
      z = np.dot(params['w' + str(l)], values[l-1]) + params['b' + str(l)]
      a = eval(self.activation)(z)
      values.append(a)
    z = np.dot(params['w' + str(n_layers)], values[n_layers - 1]) + params['b' + str(n_layers)]
    a = softmax(z)
    if x.ndim > 1 and x.shape[1] > 1:
      ans = np.argmax(a, axis = 0)
    else:
      ans = np.argmax(a)

    return ans

  def accuracy(self, X, y):
    p = self.predict(X)
    return sum(np.equal(p, np.argmax(y, axis = 0))) / y.shape[1] * 100
