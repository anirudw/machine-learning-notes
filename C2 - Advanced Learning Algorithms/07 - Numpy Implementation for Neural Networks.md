
Implementation of a neural network using **NumPy**, providing a low-level look at the mathematical operations that frameworks like TensorFlow automate.

---

## 1. Data Pre-processing: Normalization
Neural networks perform better when input features are on a similar scale. In the coffee roasting example, Temperature ($0-300^\circ$C) and Duration ($0-20$ mins) have different ranges.

* **Process:** Subtract the mean and divide by the standard deviation for each feature.
* **TensorFlow Utility:** While the lab focuses on NumPy, it uses the `tf.keras.layers.Normalization` layer to calculate these statistics and apply them.

```python
import tensorflow as tf
# Example of preparing a normalization layer
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # computes mean, variance
Xn = norm_l(X)   # normalizes data
````

---

## 2. Manual Layer Implementation (`my_dense`)

A dense layer computes the activations for all neurons in a layer using weights (W) and biases (b).

### For-Loop Implementation

This version iterates through each unit in the layer explicitly. It is useful for understanding the logic before moving to vectorized forms.


```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def my_dense(a_in, W, b):
    """
    Computes a dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1D array 
      W    (ndarray (n,j)) : Weight matrix, n features, j units
      b    (ndarray (j, )) : Bias vector, j units
    Returns
      a_out (ndarray (j,)) : j activations
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = sigmoid(z)
    return a_out
```

---

## 3. Manual Sequential Model

To build the full network, individual `dense` layers are chained together. The output of one layer becomes the input to the next.

```python
def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x, W1, b1)
    a2 = my_dense(a1, W2, b2)
    return a2
```

---

## 4. Vectorized Implementation

The for-loop implementation is inefficient for large datasets. In practice, **vectorization** is used to perform computations on entire matrices at once.

- **Matrix Multiplication:** Instead of looping through neurons, we calculate $\mathbf{Z} = \mathbf{X}\mathbf{W} + \vec{b}$ in a single operation. 
- **Efficiency:** NumPy uses highly optimized C and BLAS libraries to handle matrix math much faster than Python loops.

```python
# Vectorized version of the same logic
def my_dense_vectorized(A_in, W, B):
    """
    A_in (ndarray (m, n)) : m examples, n features
    W    (ndarray (n, j)) : n features, j units
    B    (ndarray (1, j)) : j biases
    """
    Z = np.matmul(A_in, W) + B
    return sigmoid(Z)
```
## 5. Inference and Prediction

Once forward propagation is complete, the final activation (a probability) is converted into a discrete class.

```python
def predict(X, W1, b1, W2, b2):
    # Forward prop
    m = X.shape[0]
    p = np.zeros((m, 1))
    for i in range(m):
        p[i, 0] = my_sequential(X[i], W1, b1, W2, b2)
    
    # Thresholding
    y_hat = (p >= 0.5).astype(int)
    return y_hat
```

---

## ##Tags

#neural-networks #deep-learning #python #numpy 