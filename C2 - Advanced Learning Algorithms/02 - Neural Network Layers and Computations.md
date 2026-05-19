## The Building Block: A Layer
A neural network layer is a grouping of **neurons (units)** that take the same input vector and perform computations in parallel to output a new vector of activations.

### Anatomy of a Single Neuron Calculation
Each neuron within a layer acts as a logistic regression unit:
1.  **Input:** Receives a vector $\vec{x}$ (or the activations from the previous layer $\vec{a}^{[l-1]}$).
2.  **Parameters:** Each neuron $j$ has its own weight vector $\vec{w}_j$ and bias $b_j$.
3.  **Activation ($a$):**
    $$z_j = \vec{w}_j \cdot \vec{a}^{[l-1]} + b_j$$
    $$a_j = g(z_j) = \frac{1}{1 + e^{-z_j}}$$
    * $g(z)$ is the **activation function** (specifically the Sigmoid/Logistic function in this context).



---

## Formal Notation
As networks grow to dozens or hundreds of layers, standardized notation is used to track variables:

* **Superscript $[l]$:** Denotes the **layer number**.
* **Subscript $j$:** Denotes the **unit (neuron) number** within that layer.
* **$a_j^{[l]}$:** The activation output of the $j^{th}$ neuron in the $l^{th}$ layer.
* **$\vec{w}_j^{[l]}, b_j^{[l]}$:** The parameters (weights and bias) belonging to the $j^{th}$ neuron in the $l^{th}$ layer.

### Layer Numbering Convention
* **Layer 0:** The **Input Layer** (the raw features $\vec{x}$), also denoted as $\vec{a}^{[0]}$.
* **Layers 1 to $L-1$:** The **Hidden Layers**.
* **Layer $L$:** The **Output Layer**.

> **Note:** When a network is described as a "4-layer network," it typically refers to the number of hidden layers plus the output layer (the input layer is not counted).

---

## The General Equation for Any Layer
To calculate the activation of any unit $j$ in layer $l$:

$$a_j^{[l]} = g(\vec{w}_j^{[l]} \cdot \vec{a}^{[l-1]} + b_j^{[l]})$$

* **Input to Layer $l$:** The activation vector from the previous layer, $\vec{a}^{[l-1]}$.
* **Output of Layer $l$:** A new vector $\vec{a}^{[l]}$ containing the results of all neurons in that layer.



---

## Making Predictions (Inference)
Once the final activation $a^{[L]}$ is calculated at the output layer:
1.  **Probability:** The value $a^{[L]}$ represents the probability (e.g., $P(y=1|x)$).
2.  **Thresholding (Optional):** To get a binary classification $\hat{y}$ (0 or 1), a threshold is applied (usually **0.5**).
    * If $a^{[L]} \ge 0.5 \implies \hat{y} = 1$
    * If $a^{[L]} < 0.5 \implies \hat{y} = 0$
---
## Tags

#deep-learning  #neural-networks 