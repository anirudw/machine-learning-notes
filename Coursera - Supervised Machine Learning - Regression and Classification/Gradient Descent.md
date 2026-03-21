## 1. Objective

Minimize the [[Cost Function]]:

$$
J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2
$$

Goal:

$$
\min_{w,b} J(w,b)
$$

---

## 2. Core Idea

Gradient descent is an **iterative optimization algorithm** that updates parameters in the direction of decreasing cost.

- Move **against the gradient (slope)**
- Repeat until convergence.  

---

## 3. Update Rule

### General Form

$$
\theta := \theta - \alpha \frac{\partial J}{\partial \theta}
$$

Where:
- $\alpha$: learning rate  
- $\frac{\partial J}{\partial \theta}$: gradient  

---

## 4. Gradient Descent for Linear Regression

### Parameter Updates

$$
w := w - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)x^{(i)}
$$

$$
b := b - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)
$$

---

## 5. Algorithm: Gradient Descent

**Input:**
- $x, y$: training data  
- $w, b$: initial parameters  
- $\alpha$: learning rate  
- $num\_iters$: iterations  

**Output:**
- optimized $w, b$  

**Steps:**
1. Repeat for `num_iters`:
   2. Initialize:
      - $dj\_dw = 0$
      - $dj\_db = 0$
   3. For each training example $i$:
      - Compute prediction:
        $$
        f^{(i)} = wx^{(i)} + b
        $$
      - Compute error:
        $$
        e^{(i)} = f^{(i)} - y^{(i)}
        $$
      - Accumulate gradients:
        $$
        dj\_dw += e^{(i)} \cdot x^{(i)}
        $$
        $$
        dj\_db += e^{(i)}
        $$
   4. Average gradients:
      $$
      dj\_dw = \frac{1}{m} dj\_dw
      \quad , \quad
      dj\_db = \frac{1}{m} dj\_db
      $$
   5. Update parameters simultaneously:
      $$
      w := w - \alpha \cdot dj\_dw
      $$
      $$
      b := b - \alpha \cdot dj\_db
      $$

---

## 6.Code

```python
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        error = f_wb - y[i]
        dj_dw += error * x[i]
        dj_db += error

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    w = w_in
    b = b_in

    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

    return w, b
```

---
## Tags

#regression 
