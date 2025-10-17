import torch
import torch.nn as nn
from distributed_shampoo import (
    DistributedShampoo,
    DefaultKatieConfig,
)

# Simple model
model = nn.Linear(10, 1, bias=False)
x = torch.randn(32, 10)
y = torch.randn(32, 1)

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.01,
    preconditioner_config=DefaultKatieConfig,
    precondition_frequency=10,
    start_preconditioning_step=100,  # Start much later
    use_bias_correction=True,
)

print("Testing Katie with late start preconditioning...")
# Training loop
for step in range(20):
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()

    print(f"Step {step}: Loss = {loss.item():.6f}, Param norm: {model.weight.norm().item():.6f}")

    optimizer.step()

print("\nKatie late start test completed!")
