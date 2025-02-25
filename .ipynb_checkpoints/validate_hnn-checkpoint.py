{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}

import numpy as np
import torch
import torch.nn as nn

# The same model definition as in train_hnn.py
class HNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super(HNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

def time_derivatives(model, x):
    x = x.clone().requires_grad_(True)
    H = model(x).sum()
    gradH = torch.autograd.grad(H, x, create_graph=True)[0]
    dqdt = gradH[:, 1]   # ∂H/∂p  (treated as ∂H/∂dq)
    dpdt = -gradH[:, 0]  # -∂H/∂q
    return torch.stack([dqdt, dpdt], dim=1)

# Load the trained model
#model definition
model = HNN(input_dim=2, hidden_dim=64)

# Load only weights
checkpoint = torch.load("hnn_robot_arm.pth", weights_only=True)
model.load_state_dict(checkpoint)
model.eval()

# Load simulation data
data = np.load("robot_arm_data.npy")  # shape (N, 2) => [q, dq]
dt = 1.0 / 240.0  # same time step as training

# Let's pick an initial state from the data (e.g., the first data point)
initial_state = data[0]  # [q0, dq0]
def simulate_hnn(model, initial_state, steps=1000, dt=1.0/240.0):
    """
    Simulate the system forward using the trained HNN.
    """
    trajectory = []
    state = torch.tensor(initial_state, dtype=torch.float32).unsqueeze(0)  # shape (1,2)
    
    for _ in range(steps):
        trajectory.append(state.detach().numpy().flatten())
        # Compute derivatives [dqdt, dpdt]
        derivatives = time_derivatives(model, state)  # shape (1,2)
        # Forward Euler step: x_{n+1} = x_n + dt * dx/dt
        state = state + dt * derivatives
    
    return np.array(trajectory)

# Simulate for 1000 steps
predicted_trajectory = simulate_hnn(model, initial_state, steps=1000, dt=dt)

import matplotlib.pyplot as plt

# Extract a matching slice from the real data
real_trajectory = data[:1000]  # shape (1000,2)

# Plot q (joint angle) over time
time_axis = np.arange(1000) * dt
plt.figure(figsize=(10,4))

# Plot real data
plt.plot(time_axis, real_trajectory[:,0], label="Real q", color="blue")
# Plot HNN-predicted data
plt.plot(time_axis, predicted_trajectory[:,0], label="HNN q", color="red", linestyle="--")

plt.title("Comparison of Joint Angle (q) Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.legend()
plt.show()

import matplotlib.pyplot as plt

# Extract a matching slice from the real data
real_trajectory = data[:1000]  # shape (1000,2)

# Plot q (joint angle) over time
time_axis = np.arange(1000) * dt
plt.figure(figsize=(10,4))

# Plot real data
plt.plot(time_axis, real_trajectory[:,0], label="Real q'", color="blue")
# Plot HNN-predicted data
plt.plot(time_axis, predicted_trajectory[:,0], label="HNN q'", color="red", linestyle="--")

plt.figure(figsize=(10,4))
plt.plot(time_axis, real_trajectory[:,1], label="Real dq'", color="blue")
plt.plot(time_axis, predicted_trajectory[:,1], label="HNN dq'", color="red", linestyle="--")
plt.title("Comparison of Joint Velocity (dq') Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (rad/s)")
plt.legend()
plt.show()

