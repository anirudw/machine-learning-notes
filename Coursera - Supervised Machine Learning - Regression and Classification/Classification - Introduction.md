## Why Not Linear Regression for Classification?

- Classification output is **discrete** (e.g., 0 or 1), not continuous.
- Linear regression fits a line → predictions can go **below 0 or above 1**, which makes no sense for probabilities.
- Adding an outlier skews the decision boundary dramatically.

**Example problem:** Predict if a tumour is malignant (1) or benign (0) given its size.

---

## Logistic Regression

Despite the name, this is a **classification** algorithm.

### Sigmoid Function

Maps any real number to (0, 1):

```
g(z) = 1 / (1 + e^(-z))
```

- As z → +∞, g(z) → 1
- As z → −∞, g(z) → 0
- g(0) = 0.5

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

> **NumPy note:** `np.exp()` handles large arrays element-wise and is safe for typical float64 ranges. For very large negative z, `np.exp(-z)` → ∞, but NumPy clips this gracefully to g(z) ≈ 0.

### Logistic Regression Model

```
z = w · x + b
f(x) = g(z) = sigmoid(w · x + b)
```

- Output = **probability** that y = 1 given x, written formally as:
    
    ```
    f(x) = P(y = 1 | x; w, b)
    ```
    
- `f(x) = 0.7` → 70% chance the label is 1, 30% chance it's 0.
- Probabilities must sum to 1: `P(y=0) = 1 - P(y=1)`

---

## Decision Boundary

Threshold (usually 0.5):

```
predict y = 1  if f(x) >= 0.5  →  z >= 0  →  w·x + b >= 0
predict y = 0  if f(x) < 0.5   →  z < 0   →  w·x + b < 0
```

The **decision boundary** is the line/curve where `w·x + b = 0`.

---
## Code

```python
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Plot sigmoid
z = np.arange(-10, 11)
plt.plot(z, sigmoid(z))
plt.axhline(0.5, color='r', linestyle='--')   # decision threshold
plt.xlabel("z"); plt.ylabel("sigmoid(z)")
plt.title("Sigmoid Function")
plt.show()

# Simple prediction
def predict(x, w, b):
    z = np.dot(w, x) + b
    return sigmoid(z)
```


---
## Tags

#classification 