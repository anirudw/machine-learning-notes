 ## Setup
- `X_train`: shape `(m, n)`
- `y_train`: shape `(m,)`
- `w`: shape `(n,)`
- `b`: scalar

Example dataset:
```python
X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
```

## Logistic regression model
$$
f_{w,b}(x) = g(w \cdot x + b)
$$

where
$$
g(z) = \frac{1}{1 + e^{-z}}
$$

## Logistic loss for one example
$$
\text{loss}(f_{w,b}(x^{(i)}), y^{(i)}) =
-y^{(i)} \log(f_{w,b}(x^{(i)}))
-(1-y^{(i)}) \log(1-f_{w,b}(x^{(i)}))
$$

## Cost function
Average loss over all examples:
$$
J(w,b) = \frac{1}{m} \sum_{i=1}^{m} \text{loss}(f_{w,b}(x^{(i)}), y^{(i)})
$$

## Algorithm: compute cost
1. Initialize `cost = 0`
2. For each training example `i`:
   - Compute `z_i = np.dot(X[i], w) + b`
   - Compute `f_wb_i = sigmoid(z_i)`
   - Add logistic loss for that example
3. Divide total by `m`
4. Return cost

## Code
```python
def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0.0

    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)

    cost = cost / m
    return cost
```

## Example values
```python
w_tmp = np.array([1, 1])
b_tmp = -3
compute_cost_logistic(X_train, y_train, w_tmp, b_tmp)
```

Expected cost:
```python
0.3668667864055175
```

## Comparing parameters
Two decision boundaries:
- `b = -3`, `w = [1, 1]`
- `b = -4`, `w = [1, 1]`

The cost is lower for the better-fitting boundary.

Example:
```python
w_array1 = np.array([1, 1])
b_1 = -3

w_array2 = np.array([1, 1])
b_2 = -4

compute_cost_logistic(X_train, y_train, w_array1, b_1)
compute_cost_logistic(X_train, y_train, w_array2, b_2)
```

Expected:
```python
Cost for b = -3 : 0.3668667864055175
Cost for b = -4 : 0.5036808636748461
```

## Key idea
- Lower cost means better fit to the training labels
- Logistic loss is the per-example error
- Cost is the average loss over the full dataset

---
## Tags

#classification #logistic_regression