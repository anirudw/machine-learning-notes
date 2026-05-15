## 1. What problem softmax solves

Softmax regression generalizes logistic regression from **binary classification** to **multiclass classification**.

- Logistic regression: `y ∈ {0,1}`
- Softmax regression: `y ∈ {1,2,...,n}`

It is used when exactly **one class** is correct for each example.

---

## 2. Logistic regression recap

For binary classification:

$$
z = w \cdot x + b
$$

$$
a = g(z) = \frac{1}{1 + e^{-z}}
$$

Interpretation:

- `a` is the model’s estimate of `P(y = 1 | x)`
- `1 - a` is the model’s estimate of `P(y = 0 | x)`

---

## 3. Softmax regression idea

For `n` classes, softmax regression computes one score per class:

$$
z_j = w_j \cdot x + b_j \quad \text{for } j=1,\dots,n
$$

Then converts them into probabilities:

$$
a_j = \frac{e^{z_j}}{\sum_{k=1}^{n} e^{z_k}}
$$

Interpretation:

- `a_j` is the estimate of `P(y = j | x)`
- The outputs always satisfy:

$$
\sum_{j=1}^{n} a_j = 1
$$

---

## 4. Four-class example

If `y` can be `1,2,3,4`, then softmax computes:

$$
z_1 = w_1 \cdot x + b_1
$$

$$
z_2 = w_2 \cdot x + b_2
$$

$$
z_3 = w_3 \cdot x + b_3
$$

$$
z_4 = w_4 \cdot x + b_4
$$

and

$$
a_1 = \frac{e^{z_1}}{e^{z_1}+e^{z_2}+e^{z_3}+e^{z_4}}
$$

$$
a_2 = \frac{e^{z_2}}{e^{z_1}+e^{z_2}+e^{z_3}+e^{z_4}}
$$

$$
a_3 = \frac{e^{z_3}}{e^{z_1}+e^{z_2}+e^{z_3}+e^{z_4}}
$$

$$
a_4 = \frac{e^{z_4}}{e^{z_1}+e^{z_2}+e^{z_3}+e^{z_4}}
$$

Example:
- if `a1 = 0.30`
- `a2 = 0.20`
- `a3 = 0.15`

then

$$
a_4 = 1 - 0.30 - 0.20 - 0.15 = 0.35
$$

---

## 5. General softmax model

For `n` classes:

$$
z_j = w_j \cdot x + b_j
$$

$$
a_j = \frac{e^{z_j}}{\sum_{k=1}^{n} e^{z_k}}
$$

where `j` is the class index.

---

## 6. Relation to logistic regression

Softmax with `n = 2` reduces to logistic regression.

So softmax regression is the **multiclass generalization** of logistic regression.

---

## 7. Softmax cost function

For logistic regression, the loss can be written using `a1` and `a2`:

- if `y = 1`, loss is `-log(a1)`
- if `y = 0`, loss is `-log(a2)`

For softmax regression, the loss is:

$$
L(a, y) = -\log(a_j)
$$

where `j` is the correct class label.

So:

- if `y = 1`, loss is `-log(a1)`
- if `y = 2`, loss is `-log(a2)`
- ...
- if `y = n`, loss is `-log(an)`

Dataset cost:

$$
J = \frac{1}{m}\sum_{i=1}^{m} L\left(a^{(i)}, y^{(i)}\right)
$$

---

## 8. Why this loss works

The curve of `-log(a_j)` has this behavior:

- if `a_j` is close to `1`, loss is small
- if `a_j` is moderate, loss is larger
- if `a_j` is close to `0`, loss is very large

This pushes the model to assign high probability to the correct class.

For each training example, only the loss for the true class is used.

---

## 9. Multi-class classification with a neural network

Softmax regression can be used as the **output layer** of a neural network for multiclass classification.

Example for 10 handwritten digit classes:

- input `x`
- hidden layers
- output layer with `10` units
- softmax activation on the output layer

Forward propagation:

$$
a^{[1]} = g(W^{[1]}x + b^{[1]})
$$

$$
a^{[2]} = g(W^{[2]}a^{[1]} + b^{[2]})
$$

$$
a^{[3]} = softmax(W^{[3]}a^{[2]} + b^{[3]})
$$

The output vector contains probabilities for all classes.

---

## 10. Softmax layer behavior

Unlike sigmoid, ReLU, or linear activations, softmax is special because:

- each output depends on **all** logits `z_1 ... z_n`
- the outputs are computed jointly
- the probabilities are normalized to sum to `1`

---

## 11. TensorFlow implementation

A typical multiclass model:

```python
model = Sequential([
    Dense(25, activation='relu'),
    Dense(15, activation='relu'),
    Dense(10, activation='softmax')
])
```

For multiclass classification, TensorFlow uses:

- `SparseCategoricalCrossentropy` as the loss
- `model.fit(...)` for training
- `model.predict(...)` for inference

---

## 12. Numerical stability improvement

A more numerically stable implementation is preferred.

### Why?
Directly computing:
- exponentials
- probabilities
- logs

can create round-off error or overflow/underflow.

### Better approach
In TensorFlow, use the logits directly and let the loss function handle the softmax/cross-entropy computation internally.

For multiclass classification:

```python
model = Sequential([
    Dense(25, activation='relu'),
    Dense(15, activation='relu'),
    Dense(10, activation='linear')
])

loss = SparseCategoricalCrossentropy(from_logits=True)
```

This is more numerically accurate.

---

## 13. Key terms

- **Logits**: the raw `z_j` values before softmax
- **Softmax output**: normalized probabilities over classes
- **Sparse categorical**: each example has one correct class label
- **Cross-entropy loss**: penalizes low probability assigned to the true class

---

## 14. Main takeaways

- Softmax regression extends logistic regression to multiple classes
- It outputs a probability distribution over classes
- The probabilities sum to `1`
- The loss is `-log(probability of the correct class)`
- In neural networks, softmax is the standard output activation for multiclass classification
- Using logits with `from_logits=True` is the more stable implementation
---
## Tags

#deep-learning #softmax #neural-networks 