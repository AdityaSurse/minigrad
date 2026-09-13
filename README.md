# Minigrad

A minimal, educational deep learning framework built entirely from scratch in Python. 

This project implements the core mathematical internals of modern AI frameworks like PyTorch and TensorFlow, designed to teach how Neural Networks, Automatic Differentiation (Autograd), and Backpropagation actually work under the hood.

## Features Built
- **Custom Tensor Engine**: A `Tensor` wrapper around NumPy arrays supporting mathematical operator overloading (`+`, `-`, `*`, `/`, `@`, `**`) and broadcasting.
- **Autograd Engine**: A mathematical graph-tracking system that uses the Chain Rule and Topological Sorting to compute perfect analytical gradients for any operation via `.backward()`.
- **Neural Network Layers**: Built-in composable modules including `Linear` (fully connected) layers with Xavier-style initialization, `Sequential` containers, and non-linear activations (`ReLU`, `Sigmoid`, `Tanh`).
- **Loss Functions**: Differentiable `mse_loss` and numerically-stable `cross_entropy_loss`.
- **Optimizers**: Gradient descent optimizers including `SGD` (with momentum) and `Adam`, operating efficiently on underlying data arrays.
- **Graph Visualizer**: A `graphviz`-powered tool that mathematically diagrams the Autograd computation graph.

---

## Installation

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies (NumPy, PyTest, Matplotlib, SciKit-Learn, Graphviz)
pip install numpy pytest matplotlib scikit-learn graphviz

# (Optional) For the Graphviz visualizer to render offline images on a Mac:
# brew install graphviz
```

---

## Demos & Walkthroughs

The project was built in strict phases. You can run the following demo scripts to see each phase of the framework in action:

#### Phase 1 & 2: The Autograd Engine
Demonstrates the framework dynamically calculating derivatives for a complex calculus equation.
```bash
python demo_autograd.py
```

#### Phase 3: Neural Network Layers
Demonstrates creating a Multi-Layer Perceptron (MLP) and running a forward pass.
```bash
python demo_nn.py
```

#### Phase 4: Training & Optimizers
Demonstrates setting up a tiny Neural Network to learn the mathematical rule `y = 2x` using the Adam Optimizer.
```bash
python demo_training.py
```

#### Phase 5: Real-World Training
Trains the framework on real datasets and generates `matplotlib` loss curves in your folder!
```bash
# Train an AI to solve the non-linear XOR problem
python train_xor.py

# Train an AI to recognize Hand-Written Digits (achieves >97% accuracy)
python train_digits.py
```

#### Phase 6: Computation Graph Visualizer
Traces the Autograd history of a mathematical expression and draws a flowchart of the exact operations.
```bash
python demo_viz.py
```

---

## Running Tests

The framework includes a rigorous `pytest` suite that uses numerical finite-differences to mathematically prove that the Autograd calculus engine is flawless.

```bash
PYTHONPATH=. pytest tests/
```
