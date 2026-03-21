## Goal
- Run gradient descent on a multi-variable [[Linear Regression]] model
- See how learning rate affects convergence
- Speed up convergence using feature scaling
- Use z-score normalization

## Problem setup
Housing price prediction with 4 features:
- size (sqft)
- bedrooms
- floors
- age

Typical shapes:
- `X_train`: `(m, n)`
- `y_train`: `(m,)`
- `w`: `(n,)`
- `b`: scalar

## Model
$$
f_{w,b}(x) = w \cdot x + b
$$

For all examples:
$$
\mathbf{f} = Xw + b
$$

## [[Gradient Descent]] updates
For each parameter:
$$
w_j := w_j - \alpha \frac{\partial J(w,b)}{\partial w_j}
$$

$$
b := b - \alpha \frac{\partial J(w,b)}{\partial b}
$$

## Learning rate
- `alpha` controls the size of each update
- Too small: slow convergence
- Too large: overshooting, cost may increase, no convergence
- Good `alpha`: cost decreases smoothly

## Feature scaling
If input features are on very different scales, gradient descent can be slow.

Example:
- size may be in thousands of sqft
- bedrooms and floors are small integers
- age is another scale entirely

This makes the cost contours stretched and hard to optimize.

## Z-score normalization
For each feature:
$$
x_j^{(i)} := \frac{x_j^{(i)} - \mu_j}{\sigma_j}
$$

Where:
- `μ_j` = mean of feature `j`
- `σ_j` = standard deviation of feature `j`

## Normalization procedure
1. Compute the mean of each feature
2. Compute the standard deviation of each feature
3. Transform each feature using z-score normalization
4. Run gradient descent on the normalized data

## Why it helps
- Makes feature ranges more similar
- Cost contours become more symmetric
- Gradient descent takes more direct steps toward the minimum
- Converges faster

## Key observation
Without scaling:
- updates can zigzag or take many iterations

With scaling:
- updates are more balanced across parameters

##  gradient descent

```python
for i in range(iterations):
    f = X @ w + b
    error = f - y

    dj_dw = (X.T @ error) / m
    dj_db = np.sum(error) / m

    w -= alpha * dj_dw
    b -= alpha * dj_db
```

Reference : [[Python, NumPy, and Vectorization]]
## Useful helper formulas
Mean:
$$
\mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}
$$

Standard deviation:
$$
\sigma = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu)^2}
$$

## Key takeaway
- Learning rate controls step size
- Feature scaling improves conditioning
- Z-score normalization is the standard scaling method used here
- Scaled features make gradient descent converge faster and more reliably

---
## Tags

#regression 
