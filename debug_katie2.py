import torch
import torch.nn as nn
from distributed_shampoo import (
    DistributedShampoo,
    DefaultKatieConfig,
)

# Simple model with just one parameter for easier debugging
model = nn.Linear(10, 1, bias=False)
x = torch.randn(32, 10)
y = torch.randn(32, 1)

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.01,
    preconditioner_config=DefaultKatieConfig,
    precondition_frequency=2,
    use_bias_correction=True,
)

# Training loop
for step in range(5):
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()
    
    print(f"\n=== Step {step} ===")
    print(f"Loss: {loss.item():.4f}")
    print(f"Grad norm: {model.weight.grad.norm().item():.6f}")
    
    # Access Katie preconditioner state
    param = model.weight
    if 0 in optimizer.state[param]:
        state_dict = optimizer.state[param][0]
        if 'shampoo' in state_dict:
            katie_state = state_dict['shampoo']
            
            # Check diagonal second moment
            V = katie_state.diagonal_second_moment
            print(f"Diagonal V: min={V.min().item():.6e}, max={V.max().item():.6e}, mean={V.mean().item():.6e}")
            
            # Check factor matrices
            if hasattr(katie_state, 'factor_matrices') and len(katie_state.factor_matrices) > 0:
                for i, L in enumerate(katie_state.factor_matrices):
                    print(f"Factor matrix {i}: shape={L.shape}, norm={L.norm().item():.6e}")
            
            # Check inv factor matrices  
            if hasattr(katie_state, 'inv_factor_matrices') and len(katie_state.inv_factor_matrices) > 0:
                for i, L_inv in enumerate(katie_state.inv_factor_matrices):
                    print(f"Inv factor matrix {i}: norm={L_inv.norm().item():.6e}, max_val={L_inv.abs().max().item():.6e}")
    
    optimizer.step()
    print(f"After step - Param norm: {model.weight.norm().item():.6f}")

print("\nDebug completed!")
