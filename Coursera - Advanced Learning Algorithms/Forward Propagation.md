Forward propagation is the process of a neural network making predictions (inference) by calculating activations layer-by-layer from the input to the output.

---

## Motivating Example: Digit Recognition
* **Task:** Binary classification of handwritten digits (0 vs. 1).
* **Input ($x$):** An 8x8 pixel image.
* **Features:** 64 pixel intensity values (0 for black, 255 for white).
* **Architecture:**
    * **Input Layer (Layer 0):** 64 units ($\vec{x}$ or $\vec{a}^{[0]}$).
    * **Hidden Layer 1:** 25 units ($\vec{a}^{[1]}$).
    * **Hidden Layer 2:** 15 units ($\vec{a}^{[2]}$).
    * **Output Layer 3:** 1 unit ($\vec{a}^{[3]}$).



---

## The Computation Sequence
The algorithm "propagates" activations from left to right:

1.  **Layer 1:** $$a_j^{[1]} = g(\vec{w}_j^{[1]} \cdot \vec{x} + b_j^{[1]})$$
    *Computed for $j = 1$ to 25. Results in vector $\vec{a}^{[1]}$.*

2.  **Layer 2:**
    $$a_j^{[2]} = g(\vec{w}_j^{[2]} \cdot \vec{a}^{[1]} + b_j^{[2]})$$
    *Computed for $j = 1$ to 15. Results in vector $\vec{a}^{[2]}$.*

3.  **Layer 3 (Output):**
    $$a_1^{[3]} = g(\vec{w}_1^{[3]} \cdot \vec{a}^{[2]} + b_1^{[3]})$$
    *Results in a scalar probability $P(y=1|\vec{x})$.*

4.  **Final Prediction ($\hat{y}$):**
    $$\hat{y} = 1 \text{ if } a_1^{[3]} \geq 0.5, \text{ else } 0$$

---

## Architectural Insights
* **Information Bottleneck:** A common design choice is to have more units in the initial hidden layers and decrease the number of units as the network approaches the output layer.
* **Forward vs. Backward:**
    * **Forward Propagation:** Used for **inference** (making a prediction).
    * **Backward Propagation:** Used for **learning** (adjusting weights based on error).
* **Notation Note:** $f(x)$ is often used to represent the final output $a^{[L]}$ of the entire network.
---
## Tags

#deep-learning #neural-networks 