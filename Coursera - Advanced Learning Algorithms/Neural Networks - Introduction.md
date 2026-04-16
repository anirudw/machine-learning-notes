## Artificial Neuron (Mathematical Model)
A simplified model where a neuron takes numerical inputs, performs a computation, and outputs a number (activation).
* **Activation ($a$):** The output of an individual neuron, passed to downstream neurons.

---

## Why Deep Learning is Taking Off
The performance of learning algorithms depends on the scale of data and the model size.

### The Scaling Law
* **Traditional Algorithms:** (e.g., Logistic/Linear Regression) Performance plateaus as data volume increases. They cannot effectively utilize "Big Data."
* **Neural Networks:** * **Small NNs:** Better than traditional models but still plateau.
    * **Large NNs:** Performance continues to scale with more data.
* **Enablers:** 1.  **Data:** Digitalization (Internet, mobile, IoT) has created massive datasets.
    2.  **Hardware:** Rise of **GPUs** (Graphics Processing Units), originally for graphics, now essential for parallelizing NN computations.



---

## Neural Network Architecture
A neural network consists of layers of neurons wired together.

### Layer Types
1.  **Input Layer:** The vector of input features $x$.
2.  **Hidden Layer(s):** Intermediate layers between input and output. 
    * Called "hidden" because the values are not observed in the training set (unlike $x$ and $y$).
    * A layer is a grouping of neurons that takes the same input vector and outputs a new vector of activations.
3.  **Output Layer:** The final layer that produces the prediction (e.g., probability $y$).



### Mathematical Logic
Each neuron in a layer typically performs a logistic regression function:
$$z = w \cdot x + b$$
$$a = \sigma(z) = \frac{1}{1 + e^{-z}}$$

In modern implementations, every neuron in a hidden layer usually receives inputs from **every** node in the previous layer. The model learns which features to prioritize by adjusting the weights $w$.

---

## Key Intuition: Automated Feature Engineering
* **Traditional ML:** Requires manual "feature engineering" (e.g., manually combining $x_1$ and $x_2$).
* **Neural Networks:** The hidden layers learn to create their own features.
    * *Example:* In demand prediction, a hidden layer might learn to calculate "Affordability" or "Quality" from raw inputs like price and material without explicit programming.
* **Architecture Choice:** Defining the number of hidden layers and neurons per layer is a critical design decision that impacts performance.
---
## Tags

#deep-learning #neural-networks 