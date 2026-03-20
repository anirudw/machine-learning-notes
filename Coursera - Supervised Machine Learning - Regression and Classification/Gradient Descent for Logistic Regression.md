## Model
$$
f_{w,b}(x) = g(w \cdot x + b)
$$

where
$$
g(z) = \frac{1}{1 + e^{-z}}
$$

## Logistic cost
For one example:
$$
L(f_{w,b}(x^{(i)}), y^{(i)}) =
-y^{(i)} \log(f_{w,b}(x^{(i)}))
-(1-y^{(i)}) \log(1-f_{w,b}(x^{(i)}))
$$

For the dataset:
$$
J(w,b) = \frac{1}{m} \sum_{i=1}^{m} L(f_{w,b}(x^{(i)}), y^{(i)})
$$

## Gradient
For each feature `j`:
$$
\frac{\partial J}{\partial w_j}
=
\frac{1}{m} \sum_{i=1}^{m} \left(f_{w,b}(x^{(i)}) - y^{(i)}\right) x_j^{(i)}
$$

For bias:
$$
\frac{\partial J}{\partial b}
=
\frac{1}{m} \sum_{i=1}^{m} \left(f_{w,b}(x^{(i)}) - y^{(i)}\right)
$$

## Gradient descent update
$$
w_j := w_j - \alpha \frac{\partial J}{\partial w_j}
$$

$$
b := b - \alpha \frac{\partial J}{\partial b}
$$

## Algorithm: gradient descent
1. Initialize `w` and `b`
2. Repeat for `num_iters`:
   - Compute predictions for all examples
   - Compute the error `f_wb - y`
   - Compute `dj_dw` and `dj_db`
   - Update `w` and `b`
   - Store the cost value
3. Return final `w`, `b`, and cost history

## Code: sigmoid
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

## Code: gradient computation
```python
def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        err_i = f_wb_i - y[i]
        dj_db += err_i
        for j in range(n):
            dj_dw[j] += err_i * X[i, j]

    dj_dw /= m
    dj_db /= m
    return dj_db, dj_dw
```

## Code: gradient descent
```python
def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    w = w_in.copy()
    b = b_in
    J_history = []

    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(X, y, w, b)
        J_history.append(cost)

    return w, b, J_history
```

## Dataset shape
- `X`: `(m, n)`
- `y`: `(m,)`
- `w`: `(n,)`
- `b`: scalar

## What to check
- Cost should decrease over iterations
- Parameters should move the decision boundary toward a better fit
- A reasonable learning rate is essential

---
## Tags

#classification #logistic_regression 