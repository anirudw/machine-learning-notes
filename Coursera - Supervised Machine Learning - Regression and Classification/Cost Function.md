## 1. Model Recap

$$
f_{w,b}(x) = wx + b
$$

- $f_{w,b}(x)$: prediction  
- $w, b$: parameters  
- $x$: input feature  

---

## 2. Objective

Measure how well the model fits the data.

**Goal**:
Find $w, b$ such that:

$$
f_{w,b}(x^{(i)}) \approx y^{(i)}
$$

---

## 3. Cost Function (Mean Squared Error)

$$
J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2
$$

Where:
- $m$: number of training examples  
- $f_{w,b}(x^{(i)})$: predicted value  
- $y^{(i)}$: actual value  

This computes the **average squared error** between predictions and targets.

---

## 4. Expanded Form

Substitute model:

$$
J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( wx^{(i)} + b - y^{(i)} \right)^2
$$

---

## 5. Algorithm: Compute Cost

**Input:**
- $x$: feature vector  
- $y$: target vector  
- $w, b$: parameters  

**Output:**
- cost $J(w,b)$  

**Steps:**
1. Initialize `cost_sum = 0`
2. For each training example $i$:
   - Compute prediction:
     $$
     f^{(i)} = wx^{(i)} + b
     $$
   - Compute error:
     $$
     e^{(i)} = f^{(i)} - y^{(i)}
     $$
   - Square error and accumulate:
     $$
     cost\_sum += (e^{(i)})^2
     $$
3. Compute final cost:
   $$
   J(w,b) = \frac{1}{2m} \cdot cost\_sum
   $$
4. Return $J$

---
## 6. Code

```python
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i])**2
        cost_sum += cost
        
    total_cost = cost_sum / (2 * m)
    return total_cost
    
```

---
## Tags

#regression #linear_regression 