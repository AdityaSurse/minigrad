import numpy as np
from minigrad.tensor import Tensor

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        """
        Recursively extracts all learnable parameters (Tensors with requires_grad=True)
        from this module and its submodules.
        """
        params = []
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor):
                if value.requires_grad:
                    params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
        return params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError

class Linear(Module):
    def __init__(self, in_features, out_features):
        """
        Initialize the Linear (fully-connected) layer.
        
        Weight Initialization:
        We scale the random weights by 1 / sqrt(in_features). 
        If we initialized weights to exactly zero, all neurons in a layer would compute the 
        exact same output, receive the exact same gradients during backpropagation, and 
        learn the exact same features. This is known as the "symmetry breaking" problem.
        
        Scaling by 1 / sqrt(in_features) keeps the variance of the outputs roughly 
        the same as the inputs, preventing gradients from vanishing or exploding 
        in deep networks (similar to Xavier/Kaiming initialization).
        """
        # Uniform random distribution centered at 0, scaled by 1/sqrt(in_features)
        limit = 1.0 / np.sqrt(in_features)
        w_data = np.random.uniform(-limit, limit, size=(in_features, out_features))
        self.weight = Tensor(w_data, label='weight', requires_grad=True)
        
        # Biases are typically initialized to 0 since symmetry breaking is handled by the weights
        b_data = np.zeros(out_features)
        self.bias = Tensor(b_data, label='bias', requires_grad=True)

    def forward(self, x):
        # Forward pass is just X @ W + b. 
        # Since these are Tensors, autograd handles the backward pass automatically!
        return x @ self.weight + self.bias

class ReLU(Module):
    def forward(self, x):
        return x.relu()

class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()

class Tanh(Module):
    def forward(self, x):
        return x.tanh()

class Sequential(Module):
    def __init__(self, *modules):
        self.modules = list(modules)
        
    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x
