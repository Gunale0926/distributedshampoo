import torch
import torch.nn as nn
from distributed_shampoo import (
    DistributedShampoo,
    KatiePreconditionerConfig,
)

# Simple model
model = nn.Linear(10, 1, bias=False)
x = torch.randn(32, 10)
y = torch.randn(32, 1)

# Use tuned hyperparameters based on our experiments
katie_config = KatiePreconditionerConfig(
    beta2=0.999,
    diagonal_epsilon=1e-8,
    kronecker_epsilon=1e-2,  # Key: larger epsilon for numerical stability
)

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.01,
    preconditioner_config=katie_config,
    precondition_frequency=2,
    use_bias_correction=True,
)

print("Testing Katie with tuned hyperparameters (kronecker_epsilon=1e-2)...")
# Training loop
for step in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()

    print(f"Step {step}: Loss = {loss.item():.6f}, Param norm: {model.weight.norm().item():.6f}")

    optimizer.step()

print("\nKatie tuned test completed successfully!")
