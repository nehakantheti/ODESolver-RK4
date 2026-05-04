import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. The Neural Network (The "Solver")
# Input: Time (t) -> Output: Position (x)
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # We use Tanh because we need non-zero 2nd derivatives (Acceleration)
        # Initializing neural network
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    # t is the time at which we need to predict x and v
    def forward(self, t):
        return self.net(t)

# 2. The Physics Engine (Calculating Loss)
# model is the model we defined and t is the time where we need to predict other params.
def physics_loss(model, t):
    # Enable gradient tracking for t (needed for derivatives)
    t.requires_grad = True
    
    # Forward pass: Get x from t
    x = model(t)
    
    # First derivative: Velocity (dx/dt)
    # create_graph=True allows us to take the derivative of this derivative later

    # gives the derivative of x wrt t
    # This is PyTorch’s manual gradient calculator.
    # output type is a tuple of tensors  ====> tuple(Tensor, ....)
    v = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    
    # Second derivative: Acceleration (dv/dt or d^2x/dt^2)
    a = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
    
    # Physical Constants (Same as C++ simulation)
    m = 1.0
    k = 1.0
    c = 0.1
    
    # The ODE Residual: m*a + c*v + k*x = 0
    # If the network perfectly follows physics, this should be 0.
    residual = (m * a) + (c * v) + (k * x)
    
    return torch.mean(residual ** 2)

# 3. Training Setup
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Generate "Collocation Points" (Random time points to check physics on)
# We check physics from t=0 to t=20
t_physics = torch.linspace(0, 20, 1000).view(-1, 1).to(device)

# Initial Condition Data: At t=0, x=1
t_initial = torch.tensor([[0.0]]).to(device)
x_initial = torch.tensor([[1.0]]).to(device)
# Note: Ideally we also enforce v(0)=0, but let's keep it simple first.

# 4. The Training Loop
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Loss 1: Did we match the initial condition? (Data Loss)
    x_pred_initial = model(t_initial)
    loss_ic = torch.mean((x_pred_initial - x_initial) ** 2)
    
    # Loss 2: Did we obey the laws of physics? (Physics Loss)
    loss_phy = physics_loss(model, t_physics)
    
    # Total Loss
    loss = loss_ic + loss_phy
    
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f} (IC: {loss_ic.item():.6f}, Phy: {loss_phy.item():.6f})")

# 5. Evaluation & Comparison
print("Training Complete. Generating plot...")

# Get predictions from PINN
t_test = torch.linspace(0, 20, 200).view(-1, 1).to(device)
with torch.no_grad():
    x_pred = model(t_test).cpu().numpy()

# Load C++ Ground Truth
try:
    df = pd.read_csv("results.csv", skipinitialspace=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['t'], df['rk4_x'], 'k--', label='C++ RK4 (Ground Truth)', alpha=0.7)
    plt.plot(t_test.cpu(), x_pred, 'r-', label='PINN (Neural Solver)', linewidth=2)
    
    plt.title("Comparison: Classical Solver vs Neural Solver")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (x)")
    plt.legend()
    plt.grid(True)
    plt.show()

except FileNotFoundError:
    print("Could not find results.csv. Plotting PINN only.")
    plt.plot(t_test.cpu(), x_pred, 'r-', label='PINN')
    plt.show()