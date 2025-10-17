import torch
import torch.nn as nn
from distributed_shampoo import (
    DistributedShampoo,
    DefaultShampooConfig,
)

# Simple model - same as Katie test
model = nn.Linear(10, 1, bias=False)
x = torch.randn(32, 10)
y = torch.randn(32, 1)

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.01,
    preconditioner_config=DefaultShampooConfig,
    precondition_frequency=2,
    use_bias_correction=True,
)

print("Testing baseline Shampoo...")
# Training loop
for step in range(5):
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()

    print(f"Step {step}: Loss = {loss.item():.4f}")

    optimizer.step()
    print(f"After step - Param norm: {model.weight.norm().item():.6f}")

print("\nBaseline Shampoo test completed!")
