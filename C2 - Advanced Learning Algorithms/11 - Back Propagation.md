## Goal
Backpropagation computes gradients efficiently so neural networks can learn using gradient descent.

---

# 1. Neural Network Training Flow

## Forward propagation
Compute predictions layer by layer:

$$
z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}
$$

$$
a^{[l]} = g(z^{[l]})
$$

Final output:

$$
\hat{y} = a^{[L]}
$$

---

## Compute loss

Example (binary classification):

$$
J = -\frac{1}{m}\sum_{i=1}^{m}
\left[
y^{(i)}\log(\hat{y}^{(i)})
+
(1-y^{(i)})\log(1-\hat{y}^{(i)})
\right]
$$

---

## Backpropagation
Compute gradients from output layer back to earlier layers.

Goal:

$$
\frac{\partial J}{\partial W^{[l]}}
,\quad
\frac{\partial J}{\partial b^{[l]}}
$$

Use gradients to update parameters.

---

# 2. Core Idea of Backpropagation

Backpropagation uses the **chain rule** to efficiently compute derivatives.

Instead of recomputing everything repeatedly:
- intermediate derivatives are reused
- computation becomes efficient

The computation proceeds:

```text
forward pass  → compute prediction
backward pass → compute gradients
```

citeturn0search1turn0search8

---

# 3. Output Layer Error

For the final layer:

$$
dZ^{[L]} = A^{[L]} - Y
$$

Then:

$$
dW^{[L]} =
\frac{1}{m}dZ^{[L]}(A^{[L-1]})^T
$$

$$
db^{[L]} =
\frac{1}{m}\sum dZ^{[L]}
$$

---

# 4. Backpropagation Through Hidden Layers

For hidden layer \(l\):

## Propagate error backward

$$
dZ^{[l]} =
(W^{[l+1]})^T dZ^{[l+1]}
*
g'(Z^{[l]})
$$

where:
- \(g'(z)\) = derivative of activation function
- \(*\) = elementwise multiplication

---

## Compute gradients

$$
dW^{[l]} =
\frac{1}{m}dZ^{[l]}(A^{[l-1]})^T
$$

$$
db^{[l]} =
\frac{1}{m}\sum dZ^{[l]}
$$

---

# 5. Parameter Update

Using gradient descent:

$$
W^{[l]} :=
W^{[l]} - \alpha dW^{[l]}
$$

$$
b^{[l]} :=
b^{[l]} - \alpha db^{[l]}
$$

where:
- \(\alpha\) = learning rate

---

# 6. Why Backpropagation Matters

Without backpropagation:
- training deep networks would be computationally impractical

Backpropagation:
- efficiently computes all gradients
- scales to large neural networks
- enables modern deep learning

citeturn0search8turn0search9

---

# 7. Computational Graph Intuition

Neural-network computations can be represented as a computation graph.

Example:

```text
x → z → a → J
```

Backpropagation computes derivatives from right to left:

```text
∂J/∂a
→ ∂J/∂z
→ ∂J/∂x
```

This reuse of derivatives saves computation.

citeturn0search1turn0search3

---

# 8. Activation Derivatives

## Sigmoid

$$
g(z)=\frac{1}{1+e^{-z}}
$$

Derivative:

$$
g'(z)=g(z)(1-g(z))
$$

---

## ReLU

$$
g(z)=\max(0,z)
$$

Derivative:

$$
g'(z)=
\begin{cases}
0 & z<0 \\
1 & z>0
\end{cases}
$$

---

# 9. TensorFlow Perspective

In practice, frameworks like TensorFlow compute backpropagation automatically.

Typical workflow:

```python
model.compile(...)
model.fit(X, Y)
```

TensorFlow internally:
- computes forward propagation
- computes gradients
- updates parameters

---

# 10. Main Takeaways

- Backpropagation computes gradients efficiently using the chain rule
- Gradients flow from output layer to earlier layers
- Gradients are used with gradient descent to update parameters
- Backpropagation is one of the core algorithms behind deep learning
---
## Tags

#neural-networks #deep-learning 