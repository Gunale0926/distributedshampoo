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
    precondition_frequency=2,
    use_bias_correction=True,
)

# Check what preconditioner list is being used
print(f"Preconditioner list type: {type(optimizer._per_group_state_lists[0]['shampoo_preconditioner_list'])}")
print(f"Preconditioner config: {optimizer.param_groups[0]['preconditioner_config']}")

# Training loop
for step in range(5):
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()
    
    print(f"\n=== Step {step} ===")
    print(f"Loss: {loss.item():.6f}")
    
    # Get the Katie preconditioner
    katie_precond = optimizer._per_group_state_lists[0]['shampoo_preconditioner_list']
    
    # Check bias correction values
    if hasattr(katie_precond, '_bias_correction2'):
        print(f"Kronecker bias_correction2: {katie_precond._bias_correction2.item():.6e}")
    if hasattr(katie_precond, '_bias_correction2_diagonal'):
        print(f"Diagonal bias_correction2: {katie_precond._bias_correction2_diagonal.item():.6e}")
    
    # Check if we have diagonal preconditioners
    if hasattr(katie_precond, '_local_diagonal_preconditioner_list'):
        for i, diag in enumerate(katie_precond._local_diagonal_preconditioner_list):
            print(f"Diagonal {i}: min={diag.min().item():.6e}, max={diag.max().item():.6e}, mean={diag.mean().item():.6e}")
    
    # Check Kronecker factors
    if hasattr(katie_precond, '_local_kronecker_factors_unwrapped'):
        for i, kf in enumerate(katie_precond._local_kronecker_factors_unwrapped):
            if hasattr(kf, 'inv_factor_matrices'):
                for j, inv_mat in enumerate(kf.inv_factor_matrices):
                    print(f"  Inv factor {i}.{j}: max_abs={inv_mat.abs().max().item():.6e}")
    
    optimizer.step()
    print(f"After step - Param norm: {model.weight.norm().item():.6f}")

print("\nDebug completed!")
