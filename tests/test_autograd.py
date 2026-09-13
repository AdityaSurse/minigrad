import numpy as np
import pytest
from minigrad.tensor import Tensor

def check_gradients_numerical(f, inputs, h=1e-5, tol=1e-3):
    """
    f: function that takes a list of Tensors and returns a scalar Tensor
    inputs: list of Tensors
    """
    # 1. Analytical gradients
    for t in inputs:
        t.zero_grad()
    
    out = f([t for t in inputs])
    assert out.shape == () or out.shape == (1,), "Function must return a scalar for simple grad checking"
    out.backward()
    
    # 2. Numerical gradients
    for t in inputs:
        if not t.requires_grad:
            continue
            
        grad_num = np.zeros_like(t.data)
        it = np.nditer(t.data, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_val = t.data[idx]
            
            # f(x + h)
            t.data[idx] = old_val + h
            out_plus = f(inputs).data
            if isinstance(out_plus, np.ndarray): out_plus = out_plus.item()
            
            # f(x - h)
            t.data[idx] = old_val - h
            out_minus = f(inputs).data
            if isinstance(out_minus, np.ndarray): out_minus = out_minus.item()
            
            grad_num[idx] = (out_plus - out_minus) / (2 * h)
            
            # Restore
            t.data[idx] = old_val
            it.iternext()
            
        np.testing.assert_allclose(t.grad, grad_num, atol=tol, rtol=tol)

def test_add_mul_grad():
    a = Tensor(2.0)
    b = Tensor(3.0)
    def f(inputs):
        a, b = inputs
        return a * b + a
    check_gradients_numerical(f, [a, b])

def test_pow_div_sub_grad():
    a = Tensor(2.0)
    b = Tensor(3.0)
    def f(inputs):
        a, b = inputs
        return (a ** 2) / b - a
    check_gradients_numerical(f, [a, b])

def test_broadcasting_grad():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([10.0, 20.0])
    def f(inputs):
        a, b = inputs
        c = a + b * a
        return c.sum()
    check_gradients_numerical(f, [a, b])

def test_matmul_grad():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[2.0, 0.0], [0.0, 2.0]])
    def f(inputs):
        a, b = inputs
        c = a @ b
        return c.sum()
    check_gradients_numerical(f, [a, b])

def test_activations_grad():
    # Avoid zero exactly for relu numerical derivative at 0, since it's non-differentiable there
    a = Tensor([-1.0, 0.5, 2.0])
    def f(inputs):
        a = inputs[0]
        return (a.relu() + a.sigmoid() + a.tanh() + a.exp()).sum()
    check_gradients_numerical(f, [a])

def test_log_grad():
    a = Tensor([0.5, 1.0, 2.0])
    def f(inputs):
        return inputs[0].log().sum()
    check_gradients_numerical(f, [a])

def test_complex_expression_grad():
    a = Tensor([[0.1, 0.2], [-0.3, 0.4]])
    w1 = Tensor([[0.5, -0.6], [0.7, 0.8]])
    w2 = Tensor([[0.9], [1.0]])
    # Changed b1 from [0.1, -0.1] to [0.1, -0.2] to avoid exactly 0 input to ReLU
    b1 = Tensor([0.1, -0.2])
    
    def f(inputs):
        a, w1, w2, b1 = inputs
        # typical 2 layer net forward pass
        hidden = (a @ w1 + b1).relu()
        out = hidden @ w2
        return out.sum()
        
    check_gradients_numerical(f, [a, w1, w2, b1])
