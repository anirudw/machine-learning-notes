
## Model
$$
f_{w,b}(x) = w_0x_0 + w_1x_1 + \dots + w_{n-1}x_{n-1} + b
$$

Vector form:
$$
f_{w,b}(x) = w \cdot x + b
$$

## Prediction
### Loop form
```python
def predict_single_loop(x, w, b):
    p = 0
    for i in range(x.shape[0]):
        p += x[i] * w[i]
    p += b
    return p
```

### Vectorized form
```python
def predict(x, w, b):
    return np.dot(x, w) + b
```

## Cost function
$$
J(w,b) = \frac{1}{2m} \sum_{i=0}^{m-1} \left(f_{w,b}(x^{(i)}) - y^{(i)}\right)^2
$$

```python
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i]) ** 2
    return cost / (2 * m)
```

## Gradient
For each weight `w_j`:
$$
\frac{\partial J(w,b)}{\partial w_j}
= \frac{1}{m} \sum_{i=0}^{m-1} \left(f_{w,b}(x^{(i)}) - y^{(i)}\right) x_j^{(i)}
$$

For bias:
$$
\frac{\partial J(w,b)}{\partial b}
= \frac{1}{m} \sum_{i=0}^{m-1} \left(f_{w,b}(x^{(i)}) - y^{(i)}\right)
$$

```python
def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        err = np.dot(X[i], w) + b - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err

    dj_dw /= m
    dj_db /= m
    return dj_db, dj_dw
```

## Gradient descent
Update all parameters simultaneously:
$$
w_j := w_j - \alpha \frac{\partial J(w,b)}{\partial w_j}
$$
$$
b := b - \alpha \frac{\partial J(w,b)}{\partial b}
$$

```python
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(cost_function(X, y, w, b))

    return w, b, J_history
```

## Key outputs
- Prediction for one example is a scalar
- Cost at the chosen initial parameters is extremely small
- Gradient descent reduces cost over iterations

## Important shape checks
- `X[i]` has shape `(n,)`
- `w` has shape `(n,)`
- `np.dot(X[i], w)` returns a scalar
- `X @ w` gives predictions for all examples