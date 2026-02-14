import torch
import torch.nn as nn
import time

# --- HARDWARE OPTIMIZATION 1: TF32 (The "Free Lunch") ---
# Your RTX 4060 allows "TensorFloat32" math. It's almost as fast as FP16
# but keeps the ease of use of FP32. 
torch.set_float32_matmul_precision('high')

device = torch.device("cuda")
print(f"Optimizing for: {torch.cuda.get_device_name(0)}")

# 1. Define Model
#### Smaller Model
# class PINN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(1, 32), nn.Tanh(),
#             nn.Linear(32, 32), nn.Tanh(),
#             nn.Linear(32, 1)
#         )
#     def forward(self, t): return self.net(t)

# Larger Model
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Making the network HUGE (512 width) to stress the GPU
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(), # Added depth
            nn.Linear(64, 1)
        )
    def forward(self, t) : return self.net(t)

# 2. JIT-Compatible Loss Function
# We use torch.jit.script which works on Windows (unlike torch.compile)
@torch.jit.script
def fused_physics_loss(x, t):
    # Note: JIT struggles with autograd.grad inside the script usually.
    # So we often JIT the *mathematical* part of the loss, not the gradient calls.
    # For this demo, we will JIT the forward pass logic if possible, 
    # but since Autograd is dynamic, the best speedup on Windows is 
    # simply fusing the optimizer or using TF32.
    
    # Simulating a heavy mathematical operation to show JIT benefit
    # (The ODE residual calculation itself is simple, so JIT impact is small here)
    return x

def physics_loss_pure(model, t):
    x = model(t)
    v = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    a = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
    return torch.mean((a + 0.1*v + x)**2)

# 3. Benchmark Runner
def run_benchmark(name, setup_fn, steps=2000):
    # Setup
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # STRESS TEST: Increased to 50,000 points to make the GPU sweat.
    # On small data (1000 points), CPU/Overhead dominates.
    t_physics = torch.linspace(0, 20, 50000).view(-1, 1).to(device)
    t_physics.requires_grad = True

    # Run Setup (e.g., JIT compilation)
    if setup_fn:
        model = setup_fn(model)

    # Warmup
    print(f"--- Running {name} ---")
    for _ in range(10): 
        opt_step(model, optimizer, t_physics)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(steps):
        opt_step(model, optimizer, t_physics)
        
    torch.cuda.synchronize()
    end = time.time()
    
    print(f"  Result: {end - start:.4f} seconds")

def opt_step(model, opt, t):
    opt.zero_grad()
    loss = physics_loss_pure(model, t)
    loss.backward()
    opt.step()

# --- Execution ---

# 1. Baseline (Standard Float32, No TF32)
# We temporarily disable the optimization to show the "Before" state
torch.set_float32_matmul_precision('highest') # 'highest' = slow, accurate (standard)
run_benchmark("Baseline (Standard FP32)", None)

# 2. Hardware Optimized (TF32 Enabled)
# This uses the Tensor Cores on your RTX 4060
torch.set_float32_matmul_precision('high') 
run_benchmark("Hardware Opt (TF32 Tensor Cores)", None)

# 3. JIT Scripting (Windows Safe Fusion)
def script_model(model):
    return torch.jit.script(model)

run_benchmark("JIT Scripted Model + TF32", script_model)