## Goal
- Reduce [[Overfitting]]
- Penalize large parameter values
- Improve generalization on unseen data

## Idea
Regularization adds a penalty to the cost function so the model prefers smaller weights.

## Regularized cost ([[Linear Regression]])
$$
J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
$$

- `b` is not regularized
- `\lambda` controls regularization strength

## Effect of lambda
- Small `\lambda` → weak penalty, more flexible model
- Large `\lambda` → stronger penalty, simpler model

## [[Gradient Descent]] with regularization

For `j >= 1`:
$$
w_j := w_j - \alpha \left( \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})x_j^{(i)} + \frac{\lambda}{m} w_j \right)
$$

For bias:
$$
b := b - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})
$$

## Logistic regression with regularization
$$
J(w,b) = -\frac{1}{m} \sum_{i=1}^{m} \left[y^{(i)} \log(f_{w,b}(x^{(i)})) + (1-y^{(i)}) \log(1-f_{w,b}(x^{(i)}))\right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
$$

## Why it helps
- Reduces variance
- Prevents overly large coefficients
- Makes the decision boundary smoother

## Key takeaway
- Overfitting is controlled by penalizing large weights
- Regularization keeps the model simpler
- `lambda` is the main knob for the strength of that penalty

---
## Tags

#classification #regression 