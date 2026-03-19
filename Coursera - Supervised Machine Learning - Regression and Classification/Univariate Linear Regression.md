# Model Representation

## 1. Model Definition

### Hypothesis Function

$$
f_{w,b}(x) = wx + b
$$

- $w$: weight (slope)  
- $b$: bias (intercept)  
- $x$: input feature  
- $f_{w,b}(x)$: predicted output  

---

## 2. Dataset

Given training data:

$$
x^{(i)},\ y^{(i)} \quad \text{for } i = 1,2,...,m
$$

Example:
- $x = [1.0, 2.0]$
- $y = [300.0, 500.0]$

---

## 3. Prediction Mechanism

### Algorithm: Compute Model Output

**Input:**
- Feature vector $x$
- Parameters $w, b$

**Output:**
- Predictions $f_{w,b}(x)$

**Steps:**
1. Initialize output array $f$
2. For each training example:
   - Compute prediction using model
3. Return $f$

$$
f^{(i)} = w \cdot x^{(i)} + b
$$

---

### Code

```python
def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb