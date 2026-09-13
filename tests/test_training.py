import numpy as np
import pytest
from minigrad.tensor import Tensor
from minigrad.nn import Linear
from minigrad.losses import mse_loss, cross_entropy_loss
from minigrad.optim import SGD, Adam

def check_gradients_numerical(f, inputs, h=1e-5, tol=1e-3):
    for t in inputs:
        t.zero_grad()
    
    out = f([t for t in inputs])
    out.backward()
    
    for t in inputs:
        if not t.requires_grad:
            continue
            
        grad_num = np.zeros_like(t.data)
        it = np.nditer(t.data, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_val = t.data[idx]
            
            t.data[idx] = old_val + h
            out_plus = f(inputs).data
            if isinstance(out_plus, np.ndarray): out_plus = out_plus.item()
            
            t.data[idx] = old_val - h
            out_minus = f(inputs).data
            if isinstance(out_minus, np.ndarray): out_minus = out_minus.item()
            
            grad_num[idx] = (out_plus - out_minus) / (2 * h)
            
            t.data[idx] = old_val
            it.iternext()
            
        np.testing.assert_allclose(t.grad, grad_num, atol=tol, rtol=tol)

def test_mse_loss_value():
    preds = Tensor([1.0, 2.0, 3.0])
    targets = Tensor([1.0, 2.5, 2.0])
    # squared diffs: 0.0, 0.25, 1.0 -> sum = 1.25 -> mean = 1.25 / 3 = 0.416666...
    loss = mse_loss(preds, targets)
    np.testing.assert_allclose(loss.data, 1.25 / 3)

def test_cross_entropy_loss_value():
    # 2 samples, 3 classes
    preds = Tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 1.0]])
    # one-hot targets
    targets = Tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    
    # Hand compute expected:
    # Sample 1: logits [1, 2, 3]. e_logits = [e, e^2, e^3]. sum = e+e^2+e^3 = 2.71+7.38+20.08 = 30.19
    # log_prob for class 2 = 3.0 - log(30.19) = 3.0 - 3.4076 = -0.4076
    # Sample 2: logits [3, 1, 1]. e_logits = [e^3, e, e]. sum = 20.08+2.71+2.71 = 25.5
    # log_prob for class 0 = 3.0 - log(25.5) = 3.0 - 3.2386 = -0.2386
    # NLL = (0.4076 + 0.2386) / 2 = 0.3231
    loss = cross_entropy_loss(preds, targets)
    
    # Calculate exact with numpy
    preds_np = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 1.0]])
    shifted = preds_np - np.max(preds_np, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    targets_np = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    expected_loss = -np.sum(targets_np * log_probs) / 2
    
    np.testing.assert_allclose(loss.data, expected_loss)

def test_mse_loss_grad():
    preds = Tensor([[0.1, 0.2], [-0.3, 0.4]])
    targets = Tensor([[0.0, 0.5], [-0.1, 0.4]], requires_grad=False)
    
    def f(inputs):
        return mse_loss(inputs[0], targets)
    
    check_gradients_numerical(f, [preds])

def test_cross_entropy_loss_grad():
    preds = Tensor([[0.1, 0.2, 0.7], [-0.3, 0.8, 0.4]])
    targets = Tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], requires_grad=False)
    
    def f(inputs):
        return cross_entropy_loss(inputs[0], targets)
    
    check_gradients_numerical(f, [preds])

def test_sgd_reduces_loss():
    layer = Linear(2, 1)
    # Set deterministic weights
    layer.weight.data = np.array([[1.0], [2.0]])
    layer.bias.data = np.array([0.5])
    
    x = Tensor([[0.5, -0.5]])
    target = Tensor([[1.0]])
    
    # Forward 1
    out1 = layer(x)
    loss1 = mse_loss(out1, target)
    
    # Backward & Step
    layer.zero_grad()
    loss1.backward()
    
    optim = SGD(layer.parameters(), lr=0.1)
    optim.step()
    
    # Forward 2
    out2 = layer(x)
    loss2 = mse_loss(out2, target)
    
    # Loss should decrease
    assert loss2.data < loss1.data

def test_adam_reduces_loss():
    layer = Linear(2, 1)
    layer.weight.data = np.array([[1.0], [2.0]])
    layer.bias.data = np.array([0.5])
    
    x = Tensor([[0.5, -0.5]])
    target = Tensor([[1.0]])
    
    # Forward 1
    out1 = layer(x)
    loss1 = mse_loss(out1, target)
    
    layer.zero_grad()
    loss1.backward()
    
    optim = Adam(layer.parameters(), lr=0.1)
    optim.step()
    
    # Forward 2
    out2 = layer(x)
    loss2 = mse_loss(out2, target)
    
    assert loss2.data < loss1.data

def test_zero_grad_works():
    layer = Linear(2, 1)
    x = Tensor([[0.5, 0.5]])
    out = layer(x)
    loss = out.sum()
    
    loss.backward()
    
    # Check populated
    for p in layer.parameters():
        assert p.grad is not None
        assert np.any(p.grad != 0.0)
        
    optim = SGD(layer.parameters())
    optim.zero_grad()
    
    # Check zeroed
    for p in layer.parameters():
        assert np.all(p.grad == 0.0)
