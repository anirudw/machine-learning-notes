## Goal
- Understand why squared error is not a good loss for logistic regression
- Use logistic loss for binary classification

## Logistic regression model
$$
f_{w,b}(x) = g(w \cdot x + b)
$$

where
$$
g(z) = \frac{1}{1 + e^{-z}}
$$

## Prediction
- `f(x)` is a probability-like output in `[0, 1]`
- Predict class `1` if `f(x) >= 0.5`
- Predict class `0` if `f(x) < 0.5`

## Logistic loss for one example
If `y = 1`:
$$
L(f(x), y) = -\log(f(x))
$$

If `y = 0`:
$$
L(f(x), y) = -\log(1 - f(x))
$$

Combined form:
$$
L(f(x), y) = -\left[y \log(f(x)) + (1-y) \log(1-f(x))\right]
$$

## Interpretation
- Correct confident prediction → low loss
- Wrong confident prediction → high loss
- Better calibrated for classification than squared error

## Cost over all examples
$$
J(w,b) = \frac{1}{m} \sum_{i=1}^{m} L\left(f_{w,b}(x^{(i)}), y^{(i)}\right)
$$

## Key behavior
- If `y = 1`, loss decreases as `f(x)` approaches `1`
- If `y = 0`, loss decreases as `f(x)` approaches `0`

---
## Tags

#classification #logistic_regression 