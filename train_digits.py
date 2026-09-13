import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from minigrad.tensor import Tensor
from minigrad.nn import Sequential, Linear, ReLU
from minigrad.losses import cross_entropy_loss
from minigrad.optim import Adam

# 1. Dataset
digits = load_digits()
X, y = digits.data, digits.target

# Normalize X (0-16 -> 0-1)
X = X / 16.0

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode targets
def one_hot(y, num_classes=10):
    oh = np.zeros((len(y), num_classes))
    oh[np.arange(len(y)), y] = 1.0
    return oh

y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)

X_train_t = Tensor(X_train)
y_train_t = Tensor(y_train_oh, requires_grad=False)
X_test_t = Tensor(X_test)
y_test_t = Tensor(y_test_oh, requires_grad=False)

# 2. Model: MLP
model = Sequential(
    Linear(64, 32),
    ReLU(),
    Linear(32, 10)
    # Raw logits output, since cross_entropy_loss handles softmax
)

# 3. Optimizer
optim = Adam(model.parameters(), lr=0.01)

# 4. Training loop
epochs = 200
losses = []

print("Training Digits (batch gradient descent)...")
for epoch in range(epochs):
    # Forward
    preds = model(X_train_t)
    loss = cross_entropy_loss(preds, y_train_t)
    losses.append(loss.data.item())
    
    # Backward
    model.zero_grad()
    loss.backward()
    
    # Step
    optim.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss.data.item():.4f}")

# 5. Evaluate
test_preds = model(X_test_t)
# Accuracy = argmax match
pred_classes = np.argmax(test_preds.data, axis=1)
accuracy = np.mean(pred_classes == y_test)
print(f"\n--- Digits Test Accuracy: {accuracy * 100:.2f}% ---")

# Plot and save
plt.figure()
plt.plot(losses)
plt.title("Digits Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.savefig("digits_loss_curve.png")
print("Saved digits_loss_curve.png")
