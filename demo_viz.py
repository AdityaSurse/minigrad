from minigrad.tensor import Tensor
from minigrad.viz import draw_graph

print("--- Phase 6: Computation Graph Visualizer Demo ---")

# 1. Create a simple math expression: y = relu((x @ w) + b)
x = Tensor([[1.0, 2.0]], label='Input (x)', requires_grad=False)
w = Tensor([[0.5, -0.5], [1.0, 2.0]], label='Weights (w)')
b = Tensor([[0.1, 0.2]], label='Bias (b)')

hidden = x @ w
hidden.label = 'hidden'

added = hidden + b
added.label = 'added'

y = added.relu()
y.label = 'Output (y)'

# 2. Run backward pass so we can see gradients in the graph if we wanted to
# (But for now we're just visualizing the forward graph structure)
y.backward()

# 3. Draw the graph!
print("Generating graph for y = relu((x @ w) + b)...")
draw_graph(y, format='svg', filename='computation_graph')
