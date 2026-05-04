import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. The Model
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, t): return self.net(t)

# 2. Physics & Loss Logic
def get_loss(model, t_physics, t_initial, x_initial):
    # A. Initial Condition Loss (Data Loss)
    x_pred_initial = model(t_initial)
    loss_pos = torch.mean((x_pred_initial - x_initial) ** 2)

    # B. Velocity Constraint (v(0) = 0)
    # We must clear gradients on t_initial before calculating v to avoid graph retention errors
    if t_initial.grad is not None: t_initial.grad.zero_()
    t_initial.requires_grad = True
    x_at_zero = model(t_initial)
    v_at_zero = torch.autograd.grad(x_at_zero, t_initial, 
                                    torch.ones_like(x_at_zero), 
                                    create_graph=True)[0]
    loss_vel = torch.mean(v_at_zero ** 2)

    # C. Physics Loss (Residual)
    t_physics.requires_grad = True
    x = model(t_physics)
    v = torch.autograd.grad(x, t_physics, torch.ones_like(x), create_graph=True)[0]
    a = torch.autograd.grad(v, t_physics, torch.ones_like(v), create_graph=True)[0]
    
    # ODE: m*a + c*v + k*x = 0 (m=1, c=0.1, k=1)
    residual = a + 0.1*v + x
    loss_phy = torch.mean(residual ** 2)
    
    return loss_pos + loss_vel + loss_phy

# 3. Setup
model = PINN().to(device)

# L-BFGS Optimizer
# Note: lr=1.0 is standard for L-BFGS. 
# max_iter=20 means it tries 20 evaluation steps per optimizer.step()
optimizer = torch.optim.LBFGS(model.parameters(), 
                              lr=1.0, 
                              max_iter=20, 
                              max_eval=25, 
                              history_size=50,
                              tolerance_grad=1e-7, 
                              tolerance_change=1e-9,
                              line_search_fn="strong_wolfe")

# Data
t_physics = torch.linspace(0, 20, 1000).view(-1, 1).to(device)
t_initial = torch.tensor([[0.0]]).to(device)
x_initial = torch.tensor([[1.0]]).to(device)

print("Starting Pure L-BFGS Training...")

# 4. The Training Loop (Closure Style)
start_loss = 0.0
final_loss = 0.0

def closure():
    optimizer.zero_grad()
    loss = get_loss(model, t_physics, t_initial, x_initial)
    loss.backward()
    return loss

# We run 50 "steps". Since max_iter=20, this is effectively up to 1000 iterations.
for i in range(100):
    loss = optimizer.step(closure)
    if i == 0: start_loss = loss.item()
    if i % 10 == 0:
        print(f"Step {i}: Loss = {loss.item():.9f}")
    final_loss = loss.item()

print(f"Training Complete. Start Loss: {start_loss:.4f} -> Final Loss: {final_loss:.9f}")

# 5. Plotting
t_test = torch.linspace(0, 20, 200).view(-1, 1).to(device)
with torch.no_grad():
    x_pred = model(t_test).cpu().numpy()

try:
    df = pd.read_csv("results.csv", skipinitialspace=True)
    plt.figure(figsize=(10, 6))
    plt.plot(df['t'], df['rk4_x'], 'k--', label='C++ RK4 (Ground Truth)', alpha=0.7)
    plt.plot(t_test.cpu(), x_pred, 'r-', label='Pure L-BFGS (Neural Solver)', linewidth=2)
    plt.title("Pure L-BFGS Performance")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (x)")
    plt.legend()
    plt.grid(True)
    plt.show()
except FileNotFoundError:
    print("No results.csv found. Plotting prediction only.")
    plt.plot(t_test.cpu(), x_pred, 'r-')
    plt.show()