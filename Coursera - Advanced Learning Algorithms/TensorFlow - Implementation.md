## 1. Motivating Example: Coffee Roasting
To illustrate neural network inference, consider a coffee roasting process where the goal is to predict if a roast is "Good" ($y=1$) or "Bad" ($y=0$).

* **Features ($x$):** Temperature (Celsius) and Duration (minutes).
* **Decision Boundary:** Only specific combinations of temperature and duration (a "triangle" in the feature space) result in good coffee. Too high/long = overcooked; too low/short = undercooked.



---

## 2. TensorFlow Building Blocks
TensorFlow is a leading deep learning framework (alongside PyTorch) used to implement and scale these models.

### The Dense Layer
In TensorFlow, a "Dense" layer is a standard layer where every neuron is connected to every output of the previous layer.
* **Units:** The number of neurons in the layer.
* **Activation:** The function applied to the linear output ($z$).

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Defining a hidden layer with 3 neurons and sigmoid activation
layer_1 = Dense(units=3, activation='sigmoid')

# Forward pass through the layer
a1 = layer_1(x) 
```
---

## 3. Data Representation: Matrices & Tensors

A common point of confusion is how data is shaped. TensorFlow and NumPy have a bit of a "divorced parents" relationship—they are related but maintain separate houses (and data types).

### NumPy Shapes

- **1D Vector:** `np.array([200, 17])` — A simple list, no rows/columns.
- **2D Matrix (Row Vector):** `np.array([[200, 17]])` — A $1 \times 2$ matrix.
- **2D Matrix (Column Vector):** `np.array([[200], [17]])` — A $2 \times 1$ matrix.

### TensorFlow Tensors

TensorFlow was designed for massive datasets. To maintain computational efficiency, **it strictly expects data in matrices (2D arrays)**, even for a single training example.

- **Tensor:** A data type used by TensorFlow to store and process matrices efficiently.
- **Conversion:** You can convert a Tensor `a1` back to a NumPy array using `a1.numpy()` . 

---

## 4. The `Sequential` API

While you can manually pass data through layers, professional implementation uses the `Sequential` class to string layers together into a unified model.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Define the Model
model = Sequential([
    Dense(units=3, activation='sigmoid', name='layer1'),
    Dense(units=1, activation='sigmoid', name='layer2')
])

# 2. Compile & Train (Detailed in future notes)
# model.compile(...)
# model.fit(X, y)

# 3. Inference (Forward Propagation)
# X_new must be a 2D matrix
X_new = np.array([[200, 17], [150, 5]]) 
predictions = model.predict(X_new)
```

---

## 5. Key Implementation Nuances (Lab Summary)

- **Hardware Acceleration:** Tensors allow TensorFlow to run calculations on GPUs/TPUs, which is why it enforces specific data shapes.
- **Precision:** TensorFlow defaults to `float32` (32-bit floating point numbers) to balance memory usage and numerical accuracy.
- **Predict Method:** Use `model.predict(X)` rather than calling layers manually. It handles the internal "forward prop" logic for the entire stack.
 ---
## Tags

#deep-learning #neural-networks  #python