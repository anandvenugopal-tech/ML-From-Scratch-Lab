# Logistic Regression (From Scratch)

This folder contains a full **from-scratch implementation** of Logistic Regression,  
built without using scikit-learn or any high-level ML libraries.  
It uses **Gradient Descent** as the optimizer and NumPy for numerical operations.

Logistic Regression is a fundamental algorithm for **binary classification** problems.

---

## What is Logistic Regression?

Unlike Linear Regression, which predicts continuous values,  
**Logistic Regression predicts probabilities**.

It models the probability that a given input belongs to class 1:

\[
P(y = 1 \mid x) = \sigma(Xw + b)
\]

The **sigmoid function** maps any number to a value between 0 and 1:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Based on this probability:
- If `P >= 0.5` → predict **1**
- If `P < 0.5` → predict **0**

---

## Loss Function

Logistic Regression is trained using **Binary Cross-Entropy Loss**:

\[
L = -\frac{1}{n} \sum \left[ y \log(\hat{y}) + (1-y) \log(1 - \hat{y}) \right]
\]

This loss is differentiable, enabling training via gradient descent.

---

## Files

- **`logistic_regression.py`**  
  Implements the LogisticRegression class using:
  - sigmoid activation  
  - cross-entropy loss  
  - gradient descent  

- **`README.md`**  
  Explains the theory and structure of logistic regression.

---

## Key Concepts Learned

- Classification vs regression  
- Sigmoid activation function  
- Cross-entropy loss  
- Gradient updates for classification  
- Threshold-based decision boundaries  
- Reusing the gradient descent engine  

---

## 🔗 Dependencies

This implementation uses your optimization engine:
optimization/gradient_descent.py

and mathematical background from:
math_foundations/

