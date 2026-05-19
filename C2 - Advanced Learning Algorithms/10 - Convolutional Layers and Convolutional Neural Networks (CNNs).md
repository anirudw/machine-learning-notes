

## 1. Dense Layers Recap

In a dense layer:

- every neuron receives input from **all activations** in the previous layer
- each neuron is fully connected to the previous layer

Example:

$$
a_j = g(w_j \cdot x + b_j)
$$

Dense layers alone can already build powerful neural networks.

---

# 2. Motivation for Other Layer Types

For some applications, fully connecting every neuron may be inefficient.

Problems:
- higher computation cost
- more parameters
- increased risk of overfitting
- larger training-data requirements

To address this, neural networks can use other layer types.

---

# 3. Convolutional Layer — Core Idea

Instead of looking at the **entire input**, a neuron only looks at a **small local region**.

Example with an image:
- one neuron looks only at a small patch
- another neuron looks at another patch
- different neurons specialize in different regions

So unlike dense layers:

```text
Dense Layer:
every neuron sees everything

Convolutional Layer:
each neuron sees only a local window
````

---

# 4. Why Convolutional Layers Help

## Faster computation

Each neuron processes fewer inputs.

---

## Fewer parameters

Smaller connections reduce model size.

---

## Reduced overfitting

The network becomes less likely to memorize noise.

---

## Less training data needed

Because the architecture imposes useful structure.

---

# 5. Example: Image Processing

Input:

- handwritten digit image
Instead of:
- every neuron seeing all pixels    

neurons only examine:

- local rectangular regions

Example:

- neuron 1 → top-left region
- neuron 2 → center region
- neuron 3 → bottom-right region

This allows the network to learn:

- edges
- curves
- local patterns

---

# 6. Convolutional Neural Network (CNN)

A neural network with convolutional layers is called a:

$$  
\text{Convolutional Neural Network (CNN)}  
$$

CNNs are widely used in:

- image recognition
- computer vision
- medical imaging
- object detection

---

# 7. 1D Convolution Example — ECG / EKG Signals

## ECG Signal

An ECG signal records heart activity over time.

Input:

- a sequence of numbers
    
- each value = signal amplitude at a time step
    

Example:

$$  
x_1, x_2, x_3, ..., x_{100}  
$$

Goal:

- classify whether the patient has heart disease
    

---

# 8. First Convolutional Layer

Instead of using all 100 inputs:

### Neuron 1 looks at:

$$  
x_1 \rightarrow x_{20}  
$$

### Neuron 2 looks at:

$$  
x_{11} \rightarrow x_{30}  
$$

### Neuron 3 looks at:

$$  
x_{21} \rightarrow x_{40}  
$$

and so on.

Each neuron only sees a **small window** of the signal.

This is the key characteristic of a convolutional layer.

---

# 9. Stacked Convolutional Layers

The next layer can also be convolutional.

Example:

- first hidden layer has 9 activations
    
- next layer neurons only examine subsets of those activations

Example:

- neuron examines:

$$  
a_1^{[1]} \rightarrow a_5^{[1]}  
$$

instead of all activations.

This builds hierarchical feature extraction.

---

# 10. Final Output Layer

After convolutional layers:

- a final sigmoid output neuron may combine extracted features

Example:

$$  
a = \sigma(z)  
$$

for binary classification:

- heart disease present
- heart disease absent

---

# 11. Architectural Choices in CNNs

Important design choices include:

- window size
- number of neurons
- stride / overlap
- number of convolutional layers

These architectural choices strongly affect performance.

---

# 12. Why CNNs Work Well

CNNs exploit the fact that:

- nearby inputs are often related
- local structure matters
Examples:

- nearby pixels in images
- nearby time points in ECG signals

This allows efficient feature learning.

---

# 13. Historical Note

Convolutional layers were heavily developed and popularized by:

$$  
\text{Yann LeCun}  
$$

especially for handwritten digit recognition.

---

# 14. Relation to Modern Deep Learning

Modern architectures often combine specialized layer types.

Examples:

- CNNs
- Transformers
- LSTMs
- Attention layers

A major part of neural-network research involves inventing:

- new layer types
- better architectural combinations

---

# 15. Main Takeaways

- Dense layers connect every neuron to all previous activations
- Convolutional layers use local windows instead
- CNNs are efficient for structured data like images and signals
- Convolutional layers reduce computation and overfitting
- Stacking convolutional layers enables hierarchical feature learning
---
## Tags

#neural-networks #deep-learning  #CNN