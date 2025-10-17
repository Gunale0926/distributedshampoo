import torch
import torch.nn as nn
from distributed_shampoo import (
    DistributedShampoo,
    KatiePreconditionerConfig,
)

# Simple model
def test_config(lr, precondition_freq, start_step, kronecker_eps, diagonal_eps, beta2):
    model = nn.Linear(10, 1, bias=False)
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    katie_config = KatiePreconditionerConfig(
        beta2=beta2,
        diagonal_epsilon=diagonal_eps,
        kronecker_epsilon=kronecker_eps,
    )

    optimizer = DistributedShampoo(
        model.parameters(),
        lr=lr,
        preconditioner_config=katie_config,
        precondition_frequency=precondition_freq,
        start_preconditioning_step=start_step,
        use_bias_correction=True,
    )

    losses = []
    for step in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Check for divergence
        if loss.item() > 100:
            return False, losses

    # Check if loss is decreasing
    if losses[-1] < losses[0]:
        return True, losses
    return False, losses

print("Tuning Katie hyperparameters...\n")

# Test 1: Lower learning rate
print("Test 1: Lower learning rate (0.001)")
success, losses = test_config(lr=0.001, precondition_freq=2, start_step=2,
                               kronecker_eps=1e-8, diagonal_eps=1e-8, beta2=0.999)
print(f"  Success: {success}, Final loss: {losses[-1]:.6f}\n")

# Test 2: Much lower learning rate
print("Test 2: Much lower learning rate (0.0001)")
success, losses = test_config(lr=0.0001, precondition_freq=2, start_step=2,
                               kronecker_eps=1e-8, diagonal_eps=1e-8, beta2=0.999)
print(f"  Success: {success}, Final loss: {losses[-1]:.6f}\n")

# Test 3: Larger kronecker epsilon
print("Test 3: Larger kronecker epsilon (1e-4)")
success, losses = test_config(lr=0.01, precondition_freq=2, start_step=2,
                               kronecker_eps=1e-4, diagonal_eps=1e-8, beta2=0.999)
print(f"  Success: {success}, Final loss: {losses[-1]:.6f}\n")

# Test 4: Even larger kronecker epsilon
print("Test 4: Even larger kronecker epsilon (1e-2)")
success, losses = test_config(lr=0.01, precondition_freq=2, start_step=2,
                               kronecker_eps=1e-2, diagonal_eps=1e-8, beta2=0.999)
print(f"  Success: {success}, Final loss: {losses[-1]:.6f}\n")

# Test 5: Higher beta2 for diagonal (less aggressive)
print("Test 5: Higher beta2 (0.9999) for diagonal")
success, losses = test_config(lr=0.01, precondition_freq=2, start_step=2,
                               kronecker_eps=1e-8, diagonal_eps=1e-8, beta2=0.9999)
print(f"  Success: {success}, Final loss: {losses[-1]:.6f}\n")

# Test 6: Later start step
print("Test 6: Start preconditioning at step 10")
success, losses = test_config(lr=0.01, precondition_freq=2, start_step=10,
                               kronecker_eps=1e-8, diagonal_eps=1e-8, beta2=0.999)
print(f"  Success: {success}, Final loss: {losses[-1]:.6f}\n")

# Test 7: Combination - lower LR + larger epsilon
print("Test 7: Lower LR (0.001) + larger kronecker epsilon (1e-4)")
success, losses = test_config(lr=0.001, precondition_freq=2, start_step=2,
                               kronecker_eps=1e-4, diagonal_eps=1e-8, beta2=0.999)
print(f"  Success: {success}, Final loss: {losses[-1]:.6f}\n")

# Test 8: Higher learning rate + very large epsilon
print("Test 8: LR (0.1) + very large kronecker epsilon (0.1)")
success, losses = test_config(lr=0.1, precondition_freq=2, start_step=2,
                               kronecker_eps=0.1, diagonal_eps=1e-8, beta2=0.999)
print(f"  Success: {success}, Final loss: {losses[-1]:.6f}\n")

print("Tuning completed!")
