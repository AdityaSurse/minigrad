import numpy as np
from minigrad.tensor import Tensor
from minigrad.nn import Linear, ReLU, Sequential

def test_linear_forward_shape():
    layer = Linear(in_features=3, out_features=2)
    x = Tensor(np.random.randn(4, 3)) # Batch of 4, 3 features each
    out = layer(x)
    assert out.shape == (4, 2), "Linear layer output shape is incorrect"

def test_linear_forward_computation():
    layer = Linear(2, 1)
    
    # Manually set weights and biases for predictable testing
    layer.weight.data = np.array([[1.0], [2.0]])
    layer.bias.data = np.array([0.5])
    
    # Input: 1 sample with 2 features
    x = Tensor([[0.5, -0.5]])
    
    # Expected: x @ weight + bias
    # [0.5, -0.5] @ [[1.0], [2.0]] = (0.5 * 1.0) + (-0.5 * 2.0) = 0.5 - 1.0 = -0.5
    # -0.5 + bias(0.5) = 0.0
    out = layer(x)
    
    np.testing.assert_allclose(out.data, np.array([[0.0]]), atol=1e-5)

def test_sequential_forward_shape():
    model = Sequential(
        Linear(3, 4),
        ReLU(),
        Linear(4, 2)
    )
    x = Tensor(np.random.randn(5, 3)) # Batch of 5
    out = model(x)
    
    assert out.shape == (5, 2), "Sequential model output shape is incorrect"

def test_parameters_count():
    model = Sequential(
        Linear(10, 5),
        ReLU(),
        Linear(5, 1)
    )
    params = model.parameters()
    # 2 Linear layers, each with 1 weight and 1 bias tensor = 4 total tensors
    assert len(params) == 4, "Incorrect number of parameter tensors"
    
    # Verify the actual element counts too just to be sure
    # L1: 10*5 = 50 weights + 5 biases = 55
    # L2: 5*1 = 5 weights + 1 bias = 6
    # Total: 61 scalar parameters
    total_params = sum(p.data.size for p in params)
    assert total_params == 61, "Incorrect total parameter count"

def test_gradient_flow():
    model = Sequential(
        Linear(3, 4),
        ReLU(),
        Linear(4, 2)
    )
    x = Tensor(np.random.randn(5, 3))
    out = model(x)
    
    # Create a dummy scalar loss by summing the outputs
    loss = out.sum()
    
    # Clear gradients and run backward pass
    model.zero_grad()
    loss.backward()
    
    params = model.parameters()
    assert len(params) == 4
    
    # Verify every parameter received a gradient
    for p in params:
        assert p.grad is not None, "Gradient is missing"
        assert p.grad.shape == p.shape, "Gradient shape does not match parameter shape"
        # It's practically impossible for a random input and weights to have exactly 0 gradient everywhere
        assert np.any(p.grad != 0.0), "Gradient did not flow to parameter"
