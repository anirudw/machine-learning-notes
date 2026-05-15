## 1. Why activation functions are needed

Neural networks need **non-linear activation functions** to learn complex patterns.

If every neuron uses the **linear activation function**:

$$
g(z) = z
$$

then the whole network collapses into a **linear model**, no matter how many layers it has.

---

## 2. Output-layer activation: choose based on the target `y`

### Binary classification (`y ∈ {0,1}`)
Use **sigmoid**:

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

Reason:
- output is in `[0,1]`
- can be interpreted as a probability
- natural choice for logistic-style prediction

---

### Regression with positive and negative outputs
Use **linear activation**:

$$
g(z) = z
$$

Example:
- stock price change
- values can be above or below zero

---

### Regression with only non-negative outputs
Use **ReLU**:

$$
g(z) = \max(0, z)
$$

Reason:
- output is never negative
- suitable for quantities like house prices

---

## 3. Why linear activation everywhere fails

Consider a simple network:

- input `x`
- one hidden unit with parameters `w1, b1`
- one output unit with parameters `w2, b2`

If both layers use linear activation:

### Hidden layer
$$
a_1 = g(w_1x + b_1) = w_1x + b_1
$$

### Output layer
$$
a_2 = g(w_2a_1 + b_2) = w_2a_1 + b_2
$$

Substitute `a1`:

$$
a_2 = w_2(w_1x + b_1) + b_2
$$

Expand:

$$
a_2 = (w_2w_1)x + (w_2b_1 + b_2)
$$

Define:

$$
w = w_2w_1
$$

$$
b = w_2b_1 + b_2
$$

Then:

$$
a_2 = wx + b
$$

So the network is just a **linear function of the input**.

---

## 4. General consequence

If all hidden layers use linear activation, then:

- multiple layers do **not** add expressive power
- the model is equivalent to **linear regression**
- if the output layer is sigmoid, it becomes equivalent to **logistic regression**

So stacking linear layers does not help.

---

## 5. Hidden-layer activation choice

For hidden layers, the usual default is:

$$
\text{ReLU}
$$

Why ReLU:
- faster to compute than sigmoid
- avoids saturation on both sides
- trains more efficiently in practice
- works well in most neural networks

---

## 6. Practical rules

| Task | Output activation |
|---|---|
| Binary classification | Sigmoid |
| Regression with negative values allowed | Linear |
| Regression with only non-negative values | ReLU |

| Layer type | Recommended activation |
|---|---|
| Hidden layers | ReLU |

---

## 7. Main takeaway

Use non-linear activations in hidden layers.  
If you use only linear activations, the neural network loses its ability to learn anything more powerful than a linear model.

---
## Tags

#neural-networks #deep-learning #activation