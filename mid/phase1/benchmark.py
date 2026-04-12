# TO DO

import torch
import torch.nn as nn
import time

# Define the Model (Same as before)
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, t): return self.net(t)

# The Training Function
def train_performance(device_name):
    device = torch.device(device_name)
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Create large dummy data to try and stress the GPU
    # We use 10,000 points instead of 1,000 to make it "heavier"
    t_physics = torch.linspace(0, 20, 10000).view(-1, 1).to(device)
    
    start_time = time.time()
    
    # Run 1000 fast iterations
    for _ in range(1000):
        optimizer.zero_grad()
        
        # Physics Loss Calculation
        t_physics.requires_grad = True
        x = model(t_physics)
        v = torch.autograd.grad(x, t_physics, torch.ones_like(x), create_graph=True)[0]
        a = torch.autograd.grad(v, t_physics, torch.ones_like(v), create_graph=True)[0]
        residual = a + 0.1*v + x # m=1, c=0.1, k=1
        loss = torch.mean(residual**2)
        
        loss.backward()
        optimizer.step()
        
    end_time = time.time()
    print(f"Device: {device_name} | Time: {end_time - start_time:.4f} seconds")

print("--- Starting Benchmark ---")
if torch.cuda.is_available():
    train_performance("cuda")
else:
    print("CUDA not available, skipping GPU test.")

train_performance("cpu")