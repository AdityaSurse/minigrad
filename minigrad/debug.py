from minigrad.tensor import Tensor

a = Tensor(2.0)
b = Tensor(3.0)
c = a * b + a
c.backward()
print("a.grad:", a.grad)
print("b.grad:", b.grad)
