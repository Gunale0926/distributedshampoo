import torch
import torch.nn as nn
from distributed_shampoo import (
    DistributedShampoo,
    KatiePreconditionerConfig,
    DefaultKatieConfig,
)

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)

# Create model and data
model = SimpleModel()
x = torch.randn(32, 10)
y = torch.randn(32, 1)

# Test Katie with default config
print("Testing Katie with DefaultKatieConfig...")
optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.01,
    preconditioner_config=DefaultKatieConfig,
    precondition_frequency=2,
)

# Training loop
for step in range(5):
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()
    optimizer.step()
    print(f"Step {step}: Loss = {loss.item():.4f}")

print("\nKatie test completed successfully!")

# Test with custom config
print("\nTesting Katie with custom config...")
model2 = SimpleModel()
katie_config = KatiePreconditionerConfig(
    inverse_exponent_override={2: 0.25},
    beta2=0.99,
    diagonal_epsilon=1e-8,
    kronecker_epsilon=1e-6,
)

optimizer2 = DistributedShampoo(
    model2.parameters(),
    lr=0.01,
    betas=(0.9, 0.99),
    preconditioner_config=katie_config,
    precondition_frequency=2,
)

for step in range(5):
    optimizer2.zero_grad()
    output = model2(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()
    optimizer2.step()
    print(f"Step {step}: Loss = {loss.item():.4f}")

print("\nAll tests passed!")
