from minigrad.tensor import Tensor
from minigrad.nn import Linear, ReLU, Sequential

print("--- Phase 3: Neural Network Layers Demo ---")

# 1. Create a simple Multi-Layer Perceptron (MLP) Neural Network
# It takes 3 inputs, routes them through a hidden layer of 4 neurons, and outputs 2 numbers.
model = Sequential(
    Linear(in_features=3, out_features=4),
    ReLU(),
    Linear(in_features=4, out_features=2)
)

print("\n[1] Neural Network created successfully!")
print(f"Total learnable parameter tensors: {len(model.parameters())} (2 weights and 2 biases)")

# 2. Create some dummy input data (e.g., 1 sample with 3 features)
x = Tensor([[0.5, -0.2, 0.1]])
print(f"\n[2] Input data (x): \n{x.data}")

# 3. Perform a Forward Pass
# This magically passes x through Linear -> ReLU -> Linear
output = model(x)
print(f"\n[3] Network Output prediction: \n{output.data}")

# 4. Perform a Backward Pass
# We sum the output to create a single "loss" number, then backpropagate.
print("\n[4] Running backpropagation...")
loss = output.sum()

# Always clear old gradients before a new backward pass!
model.zero_grad() 
loss.backward()

print("Backpropagation complete! Let's look at the gradients for the first layer's weights:")
first_layer_weights = model.parameters()[0]
print(first_layer_weights.grad)
print("\n(Notice how the network automatically figured out the gradients for every single layer instantly!)")
