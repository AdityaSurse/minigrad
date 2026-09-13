import numpy as np
import matplotlib.pyplot as plt
from minigrad.tensor import Tensor
from minigrad.nn import Sequential, Linear, ReLU, Sigmoid
from minigrad.losses import mse_loss
from minigrad.optim import Adam

# 1. Dataset
X = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
# XOR: 1 if inputs are different, 0 if same
Y = Tensor([[0.0], [1.0], [1.0], [0.0]])

# 2. Model: MLP with 1 hidden layer
model = Sequential(
    Linear(2, 8),
    ReLU(),
    Linear(8, 1),
    Sigmoid()
)

# 3. Optimizer
optim = Adam(model.parameters(), lr=0.1)

# 4. Training loop
epochs = 200
losses = []

print("Training XOR...")
for epoch in range(epochs):
    # Forward
    preds = model(X)
    loss = mse_loss(preds, Y)
    losses.append(loss.data.item())
    
    # Backward
    model.zero_grad()
    loss.backward()
    
    # Step
    optim.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss.data.item():.4f}")

# 5. Evaluate
preds = model(X)
print("\n--- XOR Predictions ---")
for i in range(4):
    p = preds.data[i][0]
    expected = Y.data[i][0]
    print(f"Input: {X.data[i]} | Pred: {p:.4f} -> {round(p)} | Expected: {expected}")

# Plot and save
plt.figure()
plt.plot(losses)
plt.title("XOR Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.savefig("xor_loss_curve.png")
print("\nSaved xor_loss_curve.png")
