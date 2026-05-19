## Goal
- Use feature engineering to fit non-linear patterns with linear regression
- Build polynomial features from a single input feature
- Apply feature scaling before running gradient descent on higher-order features

## Core idea
Linear regression still uses:
$$
f_{w,b}(x) = w \cdot x + b
$$
but the input features can be transformed first.

## Example 1: quadratic target
Target:
$$
y = 1 + x^2
$$

If only the original feature is used, the fit is poor.

### Feature engineering
Replace the original input with:
$$
X = x^2
$$

Then the model becomes:
$$
f_{w,b}(x) = w x^2 + b
$$

## Example 2: choose among candidate features
Candidate features:
$$
x,\; x^2,\; x^3
$$

The feature that looks most linear with respect to the target is the most useful.

## Key observation
Gradient descent adjusts weights so that useful features get larger coefficients and less useful features get smaller coefficients.

## Example 3: higher-order polynomial features
For a more complex target like:
$$
y = \cos(x/2)
$$

Construct polynomial features such as:
$$
x, x^2, x^3, \dots, x^{13}
$$

Then normalize them before training.

## Feature scaling
When polynomial features grow quickly, scaling is important.

### Z-score normalization
For each feature:
$$
x_j^{(i)} := \frac{x_j^{(i)} - \mu_j}{\sigma_j}
$$

Where:
- $\mu_j$ = mean of feature $j$
- $\sigma_j$ = standard deviation of feature $j$

## Why scaling matters
- Makes features comparable in magnitude
- Helps gradient descent converge faster
- Prevents large-scale features from dominating updates

## Typical workflow
1. Create polynomial features
2. Normalize features
3. Run gradient descent
4. Inspect the learned weights

## Common NumPy tools used
```python
np.arange(...)
np.c_[...] # columnwise concat
reshape(...)
```

## Example feature construction
```python
x = np.arange(0, 20, 1)
X = np.c_[x, x**2, x**3]
```

Higher-order example:
```python
X = np.c_[x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
```

## Main takeaway
- Feature engineering lets linear regression model non-linear data
- Polynomial features turn one input into many useful inputs
- Feature scaling is critical when polynomial terms have very different ranges
- The model remains linear in the parameters even after feature transformation

---
## Tags
#regression #polynomial_regression 