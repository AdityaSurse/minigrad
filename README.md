# Minigrad

A minimal educational deep learning framework built from scratch in Python.
This project implements the core internals of frameworks like PyTorch and TensorFlow to understand how they work under the hood.

## Features
- Basic Tensor operations with broadcasting (NumPy backed)
- Automatic Differentiation (autograd) (Coming soon)
- Simple Neural Network modules (Linear, etc.) (Coming soon)
- Optimizers (SGD, Adam) (Coming soon)
- Loss functions (MSE, Cross Entropy) (Coming soon)

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
PYTHONPATH=. pytest tests/
```
