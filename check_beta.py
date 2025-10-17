from distributed_shampoo import DefaultKatieConfig, DistributedShampoo
import torch.nn as nn

model = nn.Linear(10, 1)
opt = DistributedShampoo(
    model.parameters(),
    lr=0.01,
    preconditioner_config=DefaultKatieConfig,
)

print(f"Optimizer betas: {opt.param_groups[0]['betas']}")
print(f"Katie config beta2: {opt.param_groups[0]['preconditioner_config'].beta2}")

# Check what beta2 is passed to Katie
state_lists = opt._per_group_state_lists[0]
katie_prec = state_lists['shampoo_preconditioner_list']
print(f"Katie preconditioner _beta2: {katie_prec._beta2}")
print(f"Katie preconditioner _beta2_diagonal: {katie_prec._beta2_diagonal}")
