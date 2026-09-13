import numpy as np
from minigrad.tensor import Tensor

def test_tensor_init():
    t = Tensor(5)
    assert t.data == 5.0
    assert t.shape == ()
    
    t2 = Tensor([1, 2, 3])
    np.testing.assert_array_equal(t2.data, np.array([1, 2, 3], dtype=np.float64))
    assert t2.shape == (3,)

def test_tensor_add():
    a = Tensor([1, 2])
    b = Tensor([3, 4])
    c = a + b
    np.testing.assert_array_equal(c.data, np.array([4, 6], dtype=np.float64))
    assert c._op == '+'
    assert len(c._prev) == 2

def test_tensor_mul():
    a = Tensor([[1, 2], [3, 4]])
    b = 2
    c = a * b
    np.testing.assert_array_equal(c.data, np.array([[2, 4], [6, 8]], dtype=np.float64))

def test_tensor_sub():
    a = Tensor(5)
    b = Tensor(2)
    c = a - b
    assert c.data == 3.0
    
def test_tensor_div():
    a = Tensor(10)
    b = 2
    c = a / b
    assert c.data == 5.0

def test_tensor_pow():
    a = Tensor(2)
    c = a ** 3
    assert c.data == 8.0

def test_tensor_matmul():
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[2, 0], [0, 2]])
    c = a @ b
    np.testing.assert_array_equal(c.data, np.array([[2, 4], [6, 8]], dtype=np.float64))

def test_tensor_broadcasting():
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([10, 20])
    c = a + b
    np.testing.assert_array_equal(c.data, np.array([[11, 22], [13, 24]], dtype=np.float64))
    
def test_tensor_radd():
    a = Tensor(2)
    c = 3 + a
    assert c.data == 5.0

def test_tensor_rmul():
    a = Tensor(2)
    c = 3 * a
    assert c.data == 6.0
    
def test_tensor_neg():
    a = Tensor(5)
    c = -a
    assert c.data == -5.0
