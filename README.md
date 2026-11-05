# CS231n - Lecture 4: Two-Layer Neural Network from Scratch

This repository contains the Python implementation of a 2-layer fully-connected neural network (Multilayer Perceptron) trained from scratch using pure **NumPy**, corresponding to the practical example presented in Lecture 4 of Stanford's CS231n (*Neural Networks and Backpropagation*, Fei-Fei Li et al.)[span_0](start_span)[span_0](end_span)[span_1](start_span)[span_1](end_span)[span_2](start_span)[span_2](end_span)[span_3](start_span)[span_3](end_span)[span_4](start_span)[span_4](end_span).

---

## Code Structure & Workflow (`train_neural_net.py`)

The script simulates a complete training loop over 2000 epochs using gradient descent without relying on high-level deep learning frameworks like PyTorch or TensorFlow[span_5](start_span)[span_5](end_span).

### 1. Initialization
* **Dimensions:** Sets up a batch size $N = 64$, input dimension $D_{in} = 1000$, hidden layer size $H = 100$, and output dimension $D_{out} = 10$[span_6](start_span)[span_6](end_span).
* **Data & Weights:** Generates random input matrices ($x, y$) and initializes weight matrices ($w_1, w_2$) using standard normal distribution (`randn`)[span_7](start_span)[span_7](end_span).

### 2. Forward Pass
* **Hidden Layer Calculation:** Computes the linear transformation followed by the non-linear **Sigmoid activation function** ($\sigma(z) = \frac{1}{1 + e^{-z}}$)[span_8](start_span)[span_8](end_span)[span_9](start_span)[span_9](end_span):
  $$h = \frac{1}{1 + e^{-x \cdot w_1}}$$
* **Output Prediction:** Computes the final scores for the output layer[span_10](start_span)[span_10](end_span):
  $$y_{pred} = h \cdot w_2$$
* **Loss Evaluation:** Measures the model's error using the sum of squared differences (Euclidean loss)[span_11](start_span)[span_11](end_span):
  $$L = \sum (y_{pred} - y)^2$$

### 3. Backward Pass (Backpropagation)
The script applies the **chain rule** recursively from the output back to the input to compute exact analytical gradients for every parameter[span_12](start_span)[span_12](end_span)[span_13](start_span)[span_13](end_span):
* Calculates the gradient of the loss with respect to the predictions and propagates it through the hidden layer, factoring in the local derivative of the sigmoid function ($h \cdot (1 - h)$)[span_14](start_span)[span_14](end_span)[span_15](start_span)[span_15](end_span).
* Computes the gradients for both weight matrices (`grad_w1` and `grad_w2`) using matrix transpositions and dot products[span_16](start_span)[span_16](end_span)[span_17](start_span)[span_17](end_span).

### 4. Parameter Update (Gradient Descent)
Performs standard gradient descent by updating the weights in the opposite direction of the gradient scaled by a fixed learning rate ($\alpha = 10^{-4}$)[span_18](start_span)[span_18](end_span):
```python
w1 -= 1e-4 * grad_w1
w2 -= 1e-4 * grad_w2