from minigrad.tensor import Tensor

print("--- Phase 2: Complex Calculus Autograd Demo ---")

# Let's define a couple of variables
x = Tensor(2.0, label='x')
y = Tensor(3.0, label='y')

print(f"Given variables: x={x.data}, y={y.data}")

# Let's calculate a truly nasty mathematical equation:
# f(x, y) = e^(-(x^2 + y^2)) * tanh(x / y) + ln(x^2 + y^3 + 1.5)

print("\nEvaluating equation:")
print("f(x, y) = e^(-(x^2 + y^2)) * tanh(x / y) + ln(x^2 + y^3 + 1.5)")

# Part 1: e^(-(x^2 + y^2))
part1 = (-(x**2 + y**2)).exp()

# Part 2: tanh(x / y)
part2 = (x / y).tanh()

# Part 3: ln(x^2 + y^3 + 1.5)
part3 = ((x**2) + (y**3) + 1.5).log()

# Final function assembly
f = part1 * part2 + part3
f.label = 'f'

print(f"\nResult of f(x, y) = {f.data}")

print("\nRunning f.backward()...")
# Doing this derivative by hand on paper requires extreme patience 
# using the Chain Rule, Product Rule, and Quotient Rule.
# Our framework does it instantly.
f.backward()

print("\nThe exact derivatives (gradients) evaluated at x=2.0, y=3.0 are:")
print(f"∂f/∂x (df/dx) = {x.grad}")
print(f"∂f/∂y (df/dy) = {y.grad}")

print("\n(Try differentiating that equation by hand to verify... or just trust the framework!)")
