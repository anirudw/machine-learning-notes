
## 1. Goal

- Convert linear model output → probability  
- Enable binary classification  

---

## 2. Sigmoid Function

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

- Input: any real number  
- Output: range (0, 1)  
- S-shaped curve  


---

## 3. Key Properties

- $z \to +\infty \Rightarrow g(z) \to 1$  
- $z \to -\infty \Rightarrow g(z) \to 0$  
- $g(0) = 0.5$  

---

## 4. Logistic Regression Model

### Step 1: Linear Model

$$
z = w \cdot x + b
$$

### Step 2: Apply Sigmoid

$$
f_{w,b}(x) = g(z)
$$

---

## 5. Interpretation

$$
f_{w,b}(x) = P(y = 1 \mid x)
$$

- Output is probability of class 1  


---

## 6. Decision Rule

$$
\hat{y} =
\begin{cases}
1 & \text{if } f(x) \ge 0.5 \\
0 & \text{if } f(x) < 0.5
\end{cases}
$$

Equivalent:

$$
w \cdot x + b \ge 0
$$

---

## 7. Why Sigmoid?

- Converts linear output → probability  
- Smooth and differentiable → works with gradient descent  
- Enables classification instead of regression  

---

## 8. Visualization Insight

- Linear regression → straight line  
- Sigmoid → squashes into [0,1]  
- Threshold at 0.5 → classification boundary  

---

## 9. Key Insight

$$
\text{Linear Model} \rightarrow \text{Sigmoid} \rightarrow \text{Probability}
$$

---
## Tags

#classification #logistic_regression 