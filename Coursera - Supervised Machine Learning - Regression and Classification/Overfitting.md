# Overfitting

## Core idea
- **Overfitting** happens when a model fits the training data too closely and performs poorly on new data.
- This is typically a **high variance** problem.

## Intuition
- Simple model → may **underfit**
- Very complex model → may **overfit**
- Best model → captures the underlying trend without memorizing noise

## Polynomial regression
A linear regression model can become overly flexible by adding higher-order features:

$$
f_{w,b}(x) = w_1x + w_2x^2 + w_3x^3 + \cdots + b
$$

- Higher-degree polynomials can fit training data very well
- But too much flexibility can hurt generalization

## Symptom of overfitting
- Low training error
- High test error
- 
## Model selection idea
- Increase model complexity until performance on validation/test data stops improving
- Prefer the simplest model that generalizes well

## Bias-variance view
- **High bias** → model too simple, underfits
- **High variance** → model too complex, overfits

---
## Tags

#classification #regression 