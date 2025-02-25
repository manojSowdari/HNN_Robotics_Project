{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------
# 1) LOAD YOUR DATA
# ---------------------
data = np.load("robot_arm_data.npy")  # shape: (N, 2) or (N, 4), etc.

# If you used the sample code with a single joint, data is shape (N, 2) => [q, dq].
# For demonstration, let's assume single-joint data: [q, dq].

# Time step (dt) used in your simulation
dt = 1.0 / 240.0  # adjust if you changed it in PyBullet

# We need to compute derivatives: d(q, dq)/dt
# We'll do finite differences between consecutive time steps.
data_diff = (data[1:] - data[:-1]) / dt
data = data[:-1]  # align data points with their derivatives

# Convert to PyTorch tensors
states = torch.tensor(data, dtype=torch.float32)      # [q, dq]
states_dot = torch.tensor(data_diff, dtype=torch.float32)  # [dq/dt, ddq/dt]


# ---------------------
# 2) DEFINE THE HNN
# ---------------------
class HNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        """
        input_dim: dimension of [q, p] or [q, dq]
        hidden_dim: number of hidden units in each layer
        """
        super(HNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # outputs a single scalar: the Hamiltonian H
        )
    
    def forward(self, x):
        # x is [q, p] or [q, dq]
        return self.net(x)


def time_derivatives(model, x):
    """
    Given states x = [q, p], compute the time derivatives [dq/dt, dp/dt]
    using automatic differentiation of the Hamiltonian.
    For simplicity, treat p ~ dq, so we interpret x = [q, dq].
    """
    x = x.clone().requires_grad_(True)  # ensure we can differentiate
    H = model(x).sum()                  # sum() to handle batch gradients
    # Compute partial derivatives of H wrt x
    gradH = torch.autograd.grad(H, x, create_graph=True)[0]
    
    # For a 1D system, x = [q, dq], gradH = [∂H/∂q, ∂H/∂dq]
    dqdt = gradH[:, 1]      # ∂H/∂p (which is ∂H/∂dq)
    dpdt = -gradH[:, 0]     # -∂H/∂q
    # Combine into [dqdt, dpdt] with same shape as x
    return torch.stack([dqdt, dpdt], dim=1)


# ---------------------
# 3) TRAINING LOOP
# ---------------------
# Model, optimizer, loss
input_dim = 2  # [q, dq] for a single joint
model = HNN(input_dim=input_dim, hidden_dim=64)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epochs = 2000
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Predicted derivatives from the HNN
    pred_dot = time_derivatives(model, states)
    
    # Compare with the finite difference derivatives
    loss = loss_fn(pred_dot, states_dot)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Save the trained model
torch.save(model.state_dict(), "hnn_robot_arm.pth")
print("Training complete. Model saved to 'hnn_robot_arm.pth'")
