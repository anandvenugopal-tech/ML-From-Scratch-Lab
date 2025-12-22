# Calculus for Machine Learning

Calculus helps machine learning models **learn from mistakes**.
It tells us **how to change model parametes** so that the error becomes smaller.

In machine learning, calculus is mainly used for:
- minimizing loss functions
- training models
- updating weights using gradients

This file focuses only on **ML-relevant calculus**, explained intutively.

---

## 1. Why Calculus is Needed in Machine Learning?

Every ML model makes predictions and produces an **error (loss)**.

Example:
  Loss = (actual - predicted)^2

To improve the model, we must answer: 
> "How should the weights changes to reduce this loss?"

Calculus gives the answer.

---

## 2. Derivatives (Intution)

A **derivatives** tells us:
> how fast a value changes when another value changes.

In ML:
- derivaties of loss w.r.t weight
- it tells how the loss changes if the weight changes

If changing a weight increases loss -> move in opposite direction
If changing a weight decreases loss -> move in that direction

---

## 3. Partial Derivatives

ML models usually have **many weights**.

Example:
  L(w₁, w₂, w₃)

A **partial derivatives** measures change in loss with respect to **one weight**, keeping others fixed.

- ∂L / ∂w₁
- ∂L / ∂w₂

This allow us to update each weight seperately.

---

## 4. Gradient

The **gradient** is a vector of partial derivatives.

Example:
  ∇L = [ ∂L/∂w₁ , ∂L/∂w₂ , ∂L/∂w₃ ]

The gradient points in the direction of **steepest increase in loss**.
To reduse the loss, we move in the **opposite direction**.

---

## 5. Gradient Descent 

Gradient Descent is the algorithm that uses gradient to train models.

  w = w − α × ∂L/∂w

Where:
- `w` = weight
- `α` = learning rate
- `∂L/∂w` = derivative of loss

Meaning:
> Take a small step in the direction that reduces the loss.

---

### 6. Learning Rate (α)

The learing rate controls **step size**.

- Too small -> learning is very slow
- Too large -> model may diverge
- Proper value -> stable learning

Choosing the learning rate is crucial in ML.

---

## 7. Chain Rule (Used in Neural Networks)

The **chain rule** is used when functions are nested.

Example:
Loss -> activation -> weighted sum -> weights

The chain rule allows us to compute:
∂Loss / ∂weights

Neural networks use the chain rule repeatedly during
**backpropagation**.

---

## 9. Where Calculus Is Used in ML

- Linear Regression (MSE minimization)
- Logistic Regression (log loss)
- Neural Networks (backpropagation)
- Gradient Descent & its variants

If you understand calculus, **model training becomes clear**.

---

## Key Takeaway

> Machine learning learns by **following the slope of error**.

Calculus tells us:
- where the error increases
- where it decreases
- how to move toward better solutions
