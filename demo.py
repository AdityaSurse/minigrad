from minigrad.tensor import Tensor

# Create some tensors
a = Tensor([[1, 2], [3, 4]], label='a')
b = Tensor([[2, 0], [0, 2]], label='b')

print("Tensor A:")
print(a)
print("\nTensor B:")
print(b)

# Perform some math operations
c = a @ b  # Matrix multiplication
print("\nResult of Matrix Multiplication (A @ B):")
print(c)

d = a + 10 # Broadcasting addition
print("\nResult of Broadcasting Addition (A + 10):")
print(d)

e = a ** 2 # Power operation
print("\nResult of Power operation (A ** 2):")
print(e)
