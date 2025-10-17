import torch
import torch.nn as nn
from distributed_shampoo import (
    DistributedShampoo,
    DefaultKatieConfig,
    AdamPreconditionerConfig,
)

# Simple model
model = nn.Linear(10, 1, bias=False)
x = torch.randn(32, 10)
y = torch.randn(32, 1)

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.01,
    preconditioner_config=DefaultKatieConfig,
    precondition_frequency=2,
    use_bias_correction=True,
    grafting_config=AdamPreconditionerConfig(beta2=0.999, epsilon=1e-8),
)

print("Testing Katie with Adam grafting...")
# Training loop
for step in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()

    print(f"Step {step}: Loss = {loss.item():.6f}, Param norm: {model.weight.norm().item():.6f}")

    optimizer.step()

print("\nKatie with grafting test completed!")
