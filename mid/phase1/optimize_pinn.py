import torch
import torch.nn as nn
import time
import functools

# Check Device
device = torch.device("cuda")
print(f"Benchmarking on: {torch.cuda.get_device_name(0)}")

# 1. Define Model
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, t): return self.net(t)

# 2. Physics Loss Function (Refactored for compilation)
def physics_loss(model, t):
    x = model(t)
    # create_graph=True is required for higher-order derivatives
    v = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    a = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
    residual = a + 0.1*v + x
    return torch.mean(residual**2)

# 3. Benchmark Runner
def run_benchmark(name, train_step_func, steps=2000):
    # Setup
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    t_physics = torch.linspace(0, 20, 10000).view(-1, 1).to(device)
    t_physics.requires_grad = True

    # Warmup (critical for JIT to compile the kernels)
    print(f"--- Running {name} ---")
    print("  Warming up...")
    for _ in range(10): 
        train_step_func(model, optimizer, t_physics)
    
    # Timed Run
    torch.cuda.synchronize() # Wait for GPU to finish warmup
    start = time.time()
    
    for _ in range(steps):
        train_step_func(model, optimizer, t_physics)
        
    torch.cuda.synchronize() # Wait for GPU to finish all jobs
    end = time.time()
    
    duration = end - start
    print(f"  Result: {duration:.4f} seconds")
    return duration

# --- Strategies ---

# Strategy A: Baseline (Standard PyTorch)
def step_baseline(model, opt, t):
    opt.zero_grad()
    loss = physics_loss(model, t)
    loss.backward()
    opt.step()

# Strategy B: Mixed Precision (AMP)
scaler = torch.cuda.amp.GradScaler() # Manages the tiny gradients in float16
def step_amp(model, opt, t):
    opt.zero_grad()
    with torch.cuda.amp.autocast(): # Everything inside runs in float16 where safe
        loss = physics_loss(model, t)
    
    # Scale loss to prevent underflow, then backward
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

# Strategy C: Compiled (JIT)
# We compile the LOSS function, not just the model, to fuse the derivative math
fast_loss = torch.compile(physics_loss)

def step_compiled(model, opt, t):
    opt.zero_grad()
    loss = fast_loss(model, t)
    loss.backward()
    opt.step()

# Strategy D: Compiled + AMP (The Holy Grail)
def step_all(model, opt, t):
    opt.zero_grad()
    with torch.cuda.amp.autocast():
        loss = fast_loss(model, t)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

# --- Execution ---
try:
    base_time = run_benchmark("Baseline (FP32)", step_baseline)
    amp_time = run_benchmark("AMP (FP16)", step_amp)
    
    # Note: Compilation might take 30s+ on the very first warmup run
    print("\nNote: 'Compiled' modes may pause initially to generate kernels...")
    compile_time = run_benchmark("Torch Compile (JIT)", step_compiled)
    all_time = run_benchmark("Compile + AMP", step_all)

    print("\n--- Final Summary ---")
    print(f"Baseline:     {base_time:.4f}s")
    print(f"AMP:          {amp_time:.4f}s  (Speedup: {base_time/amp_time:.2f}x)")
    print(f"Compiled:     {compile_time:.4f}s  (Speedup: {base_time/compile_time:.2f}x)")
    print(f"Compile+AMP:  {all_time:.4f}s  (Speedup: {base_time/all_time:.2f}x)")

except Exception as e:
    print(f"\nAn error occurred (likely Windows torch.compile support): {e}")
    print("Focus on the AMP result—that is the most reliable optimization for now.")