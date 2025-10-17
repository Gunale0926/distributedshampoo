import torch
import torch.nn as nn
from distributed_shampoo import (
    DistributedShampoo,
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

# Training loop with detailed debugging
for step in range(5):
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()
    
    # Check gradient norms before step
    grad_norms = []
    for param in model.parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    
    print(f"\nStep {step}: Loss = {loss.item():.4f}, Grad norms: {grad_norms}")
    
    # Check optimizer state
    if step >= 2:  # After preconditioning starts
        for i, param in enumerate(model.parameters()):
            if 0 in optimizer.state[param]:
                state = optimizer.state[param][0]
                if 'shampoo' in state:
                    shampoo_state = state['shampoo']
                    if hasattr(shampoo_state, 'diagonal_second_moment'):
                        diag_min = shampoo_state.diagonal_second_moment.min().item()
                        diag_max = shampoo_state.diagonal_second_moment.max().item()
                        print(f"  Param {i} diagonal_second_moment: min={diag_min:.6e}, max={diag_max:.6e}")
    
    optimizer.step()
    
    # Check parameter norms after step
    param_norms = []
    for param in model.parameters():
        param_norms.append(param.norm().item())
    print(f"  After step - Param norms: {param_norms}")

print("\nDebug completed!")
