import numpy as np
from minigrad.tensor import Tensor

def mse_loss(predictions, targets):
    """
    Mean Squared Error loss.
    """
    return ((predictions - targets) ** 2).mean()

def cross_entropy_loss(predictions, targets):
    """
    Cross Entropy loss for classification.
    predictions: raw logits of shape (batch, classes)
    targets: one-hot encoded targets of shape (batch, classes)
    """
    # Numerically stable log-softmax:
    # log_probs = logits - log(sum(exp(logits)))
    # To avoid overflow, we shift logits by their max per row.
    max_logits = Tensor(np.max(predictions.data, axis=1, keepdims=True), requires_grad=False)
    shifted_logits = predictions - max_logits
    
    log_probs = shifted_logits - shifted_logits.exp().sum(axis=1, keepdims=True).log()
    
    # NLL loss: -sum(targets * log_probs) / batch_size
    loss = -(targets * log_probs).sum() / predictions.shape[0]
    return loss
