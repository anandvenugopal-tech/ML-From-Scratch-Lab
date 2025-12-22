# Probability and Statistics for Machine Learning

Probability and statistics help machine learning models **reason about data,
uncertainty, and errors**.

In machine learning, we do not use heavy theory.
We use probability and statistics to:
- summarize data
- measure uncertainty
- define loss functions
- evaluate models

This file covers **only the concepts that are actually used in ML**.

---

## 1. Why Probability and Statistics Are Important in ML?

Data is noisy and uncertain.
Machine learning models try to **learn patterns**, not memorize exact values.

Statistics helps us:
- understand data distribution
- measure spread and central tendency

Probability helps us:
- model uncertainty
- predict likelihoods
- perform classification

---

## 2. Mean (Average)

The **mean** represents the central value of data.

Formula:
 mean = (x₁ + x₂ + ... + xₙ) / n


In ML:
- used in normalization
- used in loss calculations
- used in evaluation metrics

---

## 3. Variance

**Variance** measures how spread out the data is.

Formula:
 variance = (1/n) * Σ (xᵢ − mean)²


- Low variance -> data points are close
- High variance -> data points are spread out

In ML:
- high variance models may overfit
- low variance models may underfit

---

## 4. Standard Deviation

Standard deviation is the **square root of variance**.

Why it is useful:
- same unit as data
- easier to interpret than variance

Used in:
- data scaling
- feature normalization

---

## 5. Probability

**Probability** measures how likely an event is to occur.

Range:
0 ≤ P(event) ≤ 1


In ML:
- classification models output probabilities
- uncertainty is represented probabilistically

---

## 6. Random Variables

A **random variable** maps outcomes to numerical values.

In ML:
- model predictions are random variables
- datasets are samples from unknown distributions

Understanding this helps explain why predictions are not always perfect.

---

## 7. Loss Functions

A **loss function** measures how wrong a model’s prediction is.

Common loss functions:
- Mean Squared Error (MSE) -> regression
- Log Loss (Cross-Entropy) -> classification

Lower loss means better model performance.

---

## 8. Maximum Likelihood Estimation (MLE)

MLE is a method to choose model parameters
that **maximize the probability of observing the data**.

In simple terms:
> Find parameters that best explain the data.

Many ML algorithms are based on MLE.

---

## 9. Bias–Variance Tradeoff

- **Bias**: error from overly simple models
- **Variance**: error from overly complex models

Good models balance both.

This explains why:
- too simple -> underfitting
- too complex -> overfitting

---

## 10. Where Probability & Statistics Are Used

- Data preprocessing
- Feature scaling
- Regression and classification
- Model evaluation
- Probabilistic models

Without probability and statistics, **ML cannot handle uncertainty**.

---

## Key Takeaway

> Machine learning is not about certainty.
> It is about **learning from data under uncertainty**.

Probability and statistics provide the tools to do exactly that.

