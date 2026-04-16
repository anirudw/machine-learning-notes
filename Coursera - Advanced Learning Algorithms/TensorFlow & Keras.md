
##1. Neurons as Regression Models
A single neuron in a `Dense` layer behaves exactly like a linear or logistic regression model depending on its **activation function**.

### Linear Activation (Linear Regression)
When a neuron has a `linear` activation, it calculates a weighted sum of the inputs plus a bias.
$$f_{w,b}(x^{(i)}) = w \cdot x^{(i)} + b$$

**TensorFlow Implementation:**
```python
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Defining a single neuron layer
linear_layer = Dense(units=1, activation='linear')

# Input data must be a 2D array [samples, features]
X_train = np.array([[1.0], [2.0]], dtype=np.float32)
a1 = linear_layer(X_train) 
````

### Sigmoid Activation (Logistic Regression)

When the activation is set to `sigmoid`, the neuron applies the logistic function to the linear output, mapping it to a probability between 0 and 1.

$$f_{w,b}(x^{(i)}) = g(w \cdot x^{(i)} + b)$$

**TensorFlow Implementation:**

Python

```
# A single logistic neuron
sigmoid_layer = Dense(units=1, activation='sigmoid')
a1 = sigmoid_layer(X_train)
```

---

## 2. Managing Weights and Biases

In TensorFlow, weights ($W$) and biases ($b$) are encapsulated within the layer object.

- **Initialization:** Parameters are not initialized until the layer is first called with data, as the shape of $W$ depends on the input size.
    
- **Dimensions:** For an input with $n$ features and a layer with $m$ units:
    
    - The weight matrix $W$ is shape $(n \times m)$.
        
    - The bias vector $b$ is shape $(m,)$.
        
- **Accessing Parameters:**
    

Python

```
# Extracting weights and biases
w, b = linear_layer.get_weights()

# Manually setting weights (e.g., from a pre-trained model)
linear_layer.set_weights([new_w, new_b])
```

---

## 3. Building Models with Sequential API

The `Sequential` model is the easiest way to stack layers in Keras to form a network.

Python

```
from tensorflow.keras.models import Sequential

# Constructing a simple one-layer model
model = Sequential([
    Dense(units=1, input_dim=1, activation='sigmoid', name='L1')
])

# Use summary() to inspect layers and parameter counts
model.summary()
```

---

## 4. Key Implementation Nuances

- **Input Shapes:** TensorFlow is designed for batch processing. Even for a single feature or a single example, the input must be a **2D matrix** of shape `(number_of_examples, number_of_features)`.
    
- **Tensors vs. Arrays:** While we often pass NumPy arrays, TensorFlow converts them internally into **Tensors**. You can convert a tensor back to a NumPy array using `.numpy()`.
    
- **Inference:** To run data through the model to get a prediction, use the `.predict()` method:

```python
# Predict returns the activation values (probabilities for sigmoid)
prediction = model.predict(X_train)
```

---

## Tags

#tensorflow #keras #neural-networks #deep-learning #python 