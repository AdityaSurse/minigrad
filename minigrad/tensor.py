import numpy as np

def unbroadcast(grad, target_shape):
    """
    Sums `grad` along dimensions that were broadcasted to match `target_shape`.
    """
    if grad.shape == target_shape:
        return grad
    ndims_added = grad.ndim - len(target_shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Tensor:
    def __init__(self, data, _children=(), _op='', label='', requires_grad=True):
        if isinstance(data, (int, float)):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, list):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float64) if data.dtype != np.float64 else data
        else:
            self.data = np.array(data, dtype=np.float64)
            
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape}, op={self._op}, requires_grad={self.requires_grad})"

    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = np.ones_like(self.data, dtype=np.float64)
        for v in reversed(topo):
            v._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad * other.data, self.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad * self.data, other.shape)
        out._backward = _backward
        return out
        
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad * (other * (self.data ** (other - 1))), self.shape)
        out._backward = _backward
        return out
        
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(np.matmul(self.data, other.data), (self, other), '@')
        
        def _backward():
            """
            Local backward rule for Matrix Multiplication (A @ B):
            If C = A @ B, then we want to find how changes in A and B affect the final loss L.
            By the chain rule:
                dL/dA = (dL/dC) @ B^T
                dL/dB = A^T @ (dL/dC)
            This makes sense dimensionally: 
            A is (m, n), B is (n, p), C is (m, p).
            dL/dC is (m, p).
            To get dL/dA (which must be (m, n)), we do (m, p) @ (p, n) -> (m, n).
            To get dL/dB (which must be (n, p)), we do (n, m) @ (m, p) -> (n, p).
            We use np.swapaxes(-1, -2) to transpose the last two dimensions safely,
            which handles both standard 2D matrices and batched N-D matrices correctly.
            """
            if self.requires_grad:
                grad_a = np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
                self.grad += unbroadcast(grad_a, self.shape)
            if other.requires_grad:
                grad_b = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
                other.grad += unbroadcast(grad_b, other.shape)
        out._backward = _backward
        return out
        
    def __neg__(self):
        return self * -1
        
    def __sub__(self, other):
        return self + (-other)
        
    def __truediv__(self, other):
        return self * (other ** -1)

    def __radd__(self, other):
        return self + other
        
    def __rmul__(self, other):
        return self * other
        
    def __rsub__(self, other):
        return Tensor(other, requires_grad=False) - self
        
    def __rtruediv__(self, other):
        return Tensor(other, requires_grad=False) / self

    def sum(self):
        out = Tensor(np.sum(self.data), (self,), 'sum')
        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad * np.ones_like(self.data), self.shape)
        out._backward = _backward
        return out

    # Activation functions & Math
    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,), 'relu')
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (out.data > 0)
        out._backward = _backward
        return out
        
    def sigmoid(self):
        # numerically stable sigmoid
        val = np.where(self.data >= 0, 
                       1 / (1 + np.exp(-self.data)), 
                       np.exp(self.data) / (1 + np.exp(self.data)))
        out = Tensor(val, (self,), 'sigmoid')
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (out.data * (1 - out.data))
        out._backward = _backward
        return out
        
    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,), 'tanh')
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (1 - out.data**2)
        out._backward = _backward
        return out
        
    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * out.data
        out._backward = _backward
        return out
        
    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (1 / self.data)
        out._backward = _backward
        return out
