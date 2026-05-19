## Goal 
Train neural networks faster than standard gradient descent- Adapt the effective learning rate during training---## 1. MotivationGradient descent updates parameters with a fixed learning rate:$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$$This can be slow when progress is steady in one direction, and it can oscillate when the learning rate is too large.
## 2. Why Adam helps
Adam adjusts the learning rate automatically.- 
If a parameter keeps moving in roughly the same direction, Adam increases the effective step size.- If a parameter keeps oscillating back and forth, Adam decreases the effective step size. This usually makes training faster and more stable.
## 3. What Adam means
Adam stands for **Adaptive Moment Estimation**. It uses a different effective learning rate for each parameter, rather than one global learning rate for the whole model. 
If the model has parameters:- `w1, w2, ..., wn`- `b`then Adam adapts the step size for each one separately.
## 4. Intuition
### Steady movement
If updates repeatedly point in the same direction:- increase step size- move faster toward the minimum
### Oscillating movement
If updates bounce around the minimum:- reduce step size- smooth the path to convergence---
## 5. Gradient Descent vs Adam
  
| Gradient Descent                       | Adam                                 |     |
| -------------------------------------- | ------------------------------------ | --- |
| One global learning rate               | Adaptive learning rate per parameter |     |
| More sensitive to learning-rate choice | More robust to learning-rate choice  |     |
| Can oscillate significantly            | Reduces oscillation automatically    |     |
| Often slower                           | Usually faster                       |     |
| Requires more tuning                   | Easier default behavior              |     |
## 6. TensorFlow usage
```python
model.compile(    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),    loss=...)
```
Notes:

- the model architecture stays the same
- only the optimizer changes
- `1e-3` is a common starting value
- trying a few learning rates is still useful

---

## 7. Practical recommendation

For most neural network training tasks:

- use Adam
- start with a small learning rate such as `1e-3`
- tune if needed

---

## 8. Main takeaway

Adam improves training by automatically adjusting step sizes per parameter, making learning faster and less sensitive to the exact learning rate than plain gradient descent.

---
## Tags
#neural-networks #deep-learning 