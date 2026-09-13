from minigrad.tensor import Tensor
from minigrad.nn import Linear, Sequential
from minigrad.losses import mse_loss
from minigrad.optim import Adam

print("--- Phase 4: Training Demo ---")

# 1. Create a simple AI model. 
# Since multiplying by 2 is a "linear" mathematical operation (y = 2x + 0), 
# we only need a single Linear layer. (Adding ReLUs makes the line "bend", 
# which ruins its ability to predict numbers larger than what it was trained on!)
model = Sequential(
    Linear(in_features=1, out_features=1)
)

# 2. Setup an Optimizer (Adam) to train the model's parameters
# We use a learning rate of 0.1 for faster convergence
optimizer = Adam(model.parameters(), lr=0.1)

# 3. Create a tiny dataset: We want the AI to learn to multiply numbers by 2!
x = Tensor([[1.0], [2.0], [3.0], [4.0]])
target = Tensor([[2.0], [4.0], [6.0], [8.0]])

print("Training the AI to multiply numbers by 2...\n")

# 4. A standard AI Training Loop
# We train for 500 steps so it has enough time to get the exact perfect answer.
for step in range(500):
    # Forward Pass: AI makes a guess
    predictions = model(x)
    
    # Calculate Loss: How wrong was the guess?
    loss = mse_loss(predictions, target)
    
    # Backward Pass: Calculate gradients for all weights
    optimizer.zero_grad()
    loss.backward()
    
    # Optimizer Step: Update weights to make the next guess slightly better
    optimizer.step()
    
    # Print every 50 steps so it doesn't spam the console
    if (step + 1) % 50 == 0:
        print(f"Step {step+1:03d} | Loss (Error): {loss.data.item():.6f}")

# Let's test it on a number it hasn't seen yet!
test_input = Tensor([[5.0]])
prediction = model(test_input)

print(f"\nTraining Complete! Let's test the AI.")
print(f"What is 5.0 * 2? AI predicts: {prediction.data[0][0]:.4f}")
