# Supervised Learning

This folder contains machine learning algorithms implemented **from scratch**
that learn from labeled data.  
In supervised learning, each training example has both:

- **input features (X)**
- and a **target label (y)**

The goal is to learn a function that maps inputs to outputs with minimum error.

---

## Algorithms Included

### 1. Linear Regression
- Predicts continuous values  
- Uses Gradient Descent for optimization  
- Models: `y = Xw + b`

📂 Folder: `linear_regression/`


### 2. Logistic Regression *(coming soon)*
- Binary classification  
- Uses Sigmoid + Cross-Entropy loss  
- Trained with Gradient Descent

📂 Folder: `logistic_regression/`


### 3. Perceptron *(future)*
- Basic binary classifier  
- Foundation of neural networks

📂 Folder: `perceptron/`


---

## Learning Objective

These implementations help reinforce concepts such as:

- Linear models  
- Loss functions  
- Gradient-based optimization  
- Decision boundaries  
- Training vs. inference separation  
- Clean ML code architecture  

---

## Relationship With Other Folders

- Uses optimization algorithms from:  
  **`optimization/gradient_descent.py`**

- Math concepts come from:  
  **`math_foundations/`**

---

## Goal

By completing this folder, you will understand:

✔ How supervised learning models are derived  
✔ How they learn using gradient descent  
✔ How to structure ML code like real ML frameworks

This folder forms the **core of predictive machine learning** in your project.

