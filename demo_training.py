from minigrad.tensor import Tensor
from minigrad.nn import Linear, Sequential, ReLU
from minigrad.losses import mse_loss
from minigrad.optim import Adam

print("--- Phase 4: Training Demo ---")

# 1. Create a simple AI model
model = Sequential(
    Linear(1, 8),
    ReLU(),
    Linear(8, 1)
)

# 2. Setup an Optimizer (Adam) to train the model's parameters
optimizer = Adam(model.parameters(), lr=0.1)

# 3. Create a tiny dataset: We want the AI to learn to multiply numbers by 2!
# Input x = [1, 2, 3]
# Target y = [2, 4, 6]
x = Tensor([[1.0], [2.0], [3.0]])
target = Tensor([[2.0], [4.0], [6.0]])

print("Training the AI to multiply numbers by 2...\n")

# 4. A standard AI Training Loop
for step in range(20):
    # Forward Pass: AI makes a guess
    predictions = model(x)
    
    # Calculate Loss: How wrong was the guess?
    loss = mse_loss(predictions, target)
    
    # Backward Pass: Calculate gradients for all weights
    optimizer.zero_grad()
    loss.backward()
    
    # Optimizer Step: Update weights to make the next guess slightly better
    optimizer.step()
    
    print(f"Step {step+1:02} | Loss (Error): {loss.data:.4f}")

# Let's test it on a number it hasn't seen yet!
test_input = Tensor([[5.0]])
prediction = model(test_input)
print(f"\nTraining Complete! Let's test the AI.")
print(f"What is 5.0 * 2? AI predicts: {prediction.data[0][0]:.4f}")
