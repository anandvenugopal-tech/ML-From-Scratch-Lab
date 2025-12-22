# Linear Algebra for Machine Learning

Linear algebra is the **mathematical backbone of machine learning**.
Almost every ML algorithm represents data, model parameters, and predictions
using vectors and matrices.

This document focuses only on the **linear algebra concepts that are actually
used in machine learning**, explained in simple language.

---

## 1. Scalars, Vectors, and Matrices

### Scalar
A **scalar** is a single number.

Examples:
- learning rate (α)
- bias (b)
- loss value

---

### Vector
A **vector** is a 1-dimensional array of numbers.

In ML:
- a data point (features)
- a weight vector

Example:
- x = [x₁, x₂, x₃]
- w = [w₁, w₂, w₃]


Each value in the weight vector represents how important a feature is.

---

### Matrix
A **matrix** is a 2-dimensional array of numbers.

In ML:
- the entire dataset is a matrix
- neural network layers use matrices

Example:
  X ∈ ℝ^{m×n}

where:
- 'm' = number of samples
- 'n' = number of features

---

## 2. Dot Product

The **dot product** is the most important operation in machine learning.

For vectors:
- X.W = x<sub>1</sub>w<sub>1</sub> + x<sub>2</sub>w<sub>2</sub> + ... + x<sub>n</sub>w<sub>n</sub>

In linear regression:
- ŷ = X.W + b

This means:
- multiply each feature by its weight
- add them together
- add bias

The dot measures how strongly features influence the output.

---

## 3. Matrix Multiplication

Instead of computing one by one, ML uses **matrix multiplication** to process all samples at once.

That is:
 ŷ = XW + b

where:
- 'X' = data matrix
- 'W' = weight matrix
- 'b' = bias

This makes training **fast and efficient**.

---

## 4. Why Shapes Matter

Matrix operations only work when shapes match.

Example:
- X -> (m x n)
- W -> (n x 1)
- ŷ -> (m x 1)

---

## 5. Linear Transformation

A **linear model** applies a linear transformation to data: 
- f(x) = Wx + b

This transforms input features into predictions.

Neural networks apply **multiple linear transformations** followed by non-linear activation functions.

---

## 6. Eigenvectors and Eigenvalues (used in PCA)

Eigenvectors represent **important directions** in data.
Eigenvalues tell **how much varience** exists in those directions.

In PCA:
- eigenvectors -> new axes
- eigenvalues -> importance of each axis

PCA reduces dimensions while preserving maximum informations.

---

## 7. NumPy Example

```python
import numpy as np

X = np.array([[1, 2],
              [3, 4],
              [5, 6]])

W = np.array([0.5, 1.0])
b = 0.2

y_pred = X.dot(W) + b
print(y_pred)



