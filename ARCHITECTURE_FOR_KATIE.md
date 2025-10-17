# DistributedShampoo Architecture Overview for Katie Implementation

## Executive Summary

This document provides a comprehensive overview of the DistributedShampoo optimizer architecture, focusing on the components and patterns needed to implement a DistributedKatie optimizer.

---

## 1. Main Optimizer Class

**File**: `/Users/gunale/works/claude_code/optimziation/distributedshampoo/distributed_shampoo/distributed_shampoo.py`

**Class**: `DistributedShampoo(torch.optim.Optimizer)`

### Key Architecture Points:

1. **Hierarchical State Management**:
   - `self.param_groups`: Standard PyTorch parameter groups
   - `self._per_group_state_lists`: List of dictionaries containing per-group state, indexed by parameter group
   - `self.state`: PyTorch optimizer state dict with nested structure: `self.state[param][block_index]`

2. **Distributor Pattern**:
   - Handles parameter blocking, blocking, and distributed communication
   - Instantiated per parameter group
   - State key: `DISTRIBUTOR` (stored in `_per_group_state_lists`)

3. **Preconditioner Integration**:
   - Main preconditioner: `SHAMPOO_PRECONDITIONER_LIST`
   - Grafting preconditioner: `GRAFTING_PRECONDITIONER_LIST` (optional)
   - Both are preconditioner lists that are updated and used for gradient preconditioning

4. **Auxiliary State**:
   - `STEP`: Global step counter (tensor on CPU)
   - `MOMENTUM_LIST`: Momentum buffers (if momentum > 0)
   - `FILTERED_GRAD_LIST`: EMA of gradients (if beta1 > 0)
   - `MASKED_*` variants: Filtered versions excluding None gradients

### Initialization Flow:

```python
__init__() ->
  _instantiate_distributor()          # Create Distributor instances
  _initialize_blocked_parameters_state() # Initialize block structure
  _instantiate_shampoo_preconditioner_list() # Create main preconditioner
  _instantiate_grafting()             # Create grafting preconditioner
  _instantiate_steps()                # Initialize step counter
  _instantiate_momentum()             # Initialize momentum if needed
  _instantiate_filtered_grads()       # Initialize gradient filter if needed
  _instantiate_per_group_step()       # Compile PT2 if enabled
```

### Step Loop (`step()` method):

```
For each parameter group:
  1. Construct masked gradient list from distributed gradients
  2. Mask state lists based on non-None gradients
  3. Call _per_group_step_impl() which:
     a. Compute filtered gradients
     b. Update preconditioners
     c. Apply preconditioning to gradients
     d. Apply grafting (if enabled)
     e. Apply weight decay
     f. Apply momentum
     g. Update parameters via distributor
```

---

## 2. Preconditioner Classes Hierarchy

### Base Class: `PreconditionerList` (Abstract)

**File**: `/Users/gunale/works/claude_code/optimziation/distributedshampoo/distributed_shampoo/preconditioner/preconditioner_list.py`

**Abstract Methods**:
```python
def update_preconditioners(
    self,
    masked_grad_list: tuple[Tensor, ...],
    step: Tensor,
    perform_amortized_computation: bool,
) -> None: ...

def precondition(
    self, masked_grad_list: tuple[Tensor, ...]
) -> tuple[Tensor, ...]: ...

def compress_preconditioner_list(
    self, local_grad_selector: tuple[bool, ...]
) -> None: ...
```

### Concrete Implementations:

#### 1. **AdagradPreconditionerList** 
**File**: `adagrad_preconditioner_list.py`

Implements simple diagonal preconditioning using element-wise second moment:
- State: Single diagonal tensor per parameter
- Update: `V_t = beta2 * V_{t-1} + (1-beta2) * g_t^2` (element-wise)
- Preconditioning: `g_precond = g / (sqrt(V_t) + epsilon)`

#### 2. **Shampoo Preconditioner Classes**

**Base**: `BaseShampooPreconditionerList` (Generic over Kronecker factors types)

Uses Kronecker product approximation of second moment:
- **State**: Tuple of factor matrices (one per tensor dimension)
- **Update**: Accumulates outer products
- **Key Variants**:

a. **RootInvShampooPreconditionerList**
   - Stores: Factor matrices + inverse factor matrices
   - Computation: Computes matrix inverse roots periodically
   - Preconditioning: L_inv @ G @ R_inv (left/right multiply)

b. **EigendecomposedShampooPreconditionerList**
   - Stores: Factor matrices + eigenvalues + eigenvectors
   - Computation: Eigendecomposition periodically
   - Preconditioning: Uses eigendecomposition for more stable inverse-root

c. **EigenvalueCorrectedShampooPreconditionerList** (SOAP)
   - Stores: Factor matrices + eigenvectors + corrected eigenvalues
   - Computation: Eigenvectors updated periodically, eigenvalues updated every step
   - Key difference: Eigenvalues updated in eigenbasis via gradient outer products

### Kronecker Factors State Classes

**File**: `shampoo_preconditioner_list.py`

These are stored in the optimizer state under key `SHAMPOO`:

```
BaseShampooKroneckerFactorsState (abstract)
├─ RootInvShampooKroneckerFactorsState
├─ EigendecomposedShampooKroneckerFactorsState
└─ EigenvalueCorrectedShampooKroneckerFactorsState
```

Each has both "wrapped" (stored in state) and "unwrapped" (used in computation) versions.

---

## 3. Preconditioner Configuration Classes

**File**: `/Users/gunale/works/claude_code/optimziation/distributedshampoo/distributed_shampoo/shampoo_types.py`

### Hierarchy:

```
PreconditionerConfig (abstract dataclass)
├─ SGDPreconditionerConfig
├─ AdaGradPreconditionerConfig
├─ RMSpropPreconditionerConfig
├─ AdamPreconditionerConfig
├─ AmortizedPreconditionerConfig (abstract)
│  ├─ ShampooPreconditionerConfig (abstract)
│  │  ├─ RootInvShampooPreconditionerConfig
│  │  ├─ EigendecomposedShampooPreconditionerConfig
│  │  └─ DefaultShampooConfig = RootInvShampooPreconditionerConfig()
│  └─ EigenvalueCorrectedShampooPreconditionerConfig
│     └─ DefaultEigenvalueCorrectedShampooConfig
├─ SpectralDescentPreconditionerConfig
└─ SignDescentPreconditionerConfig
```

### Key Fields in ShampooPreconditionerConfig:

```python
@dataclass(init=False)
class ShampooPreconditionerConfig(AmortizedPreconditionerConfig):
    amortized_computation_config: MatrixFunctionConfig  # How to compute inverses/eigendecomps
    num_tolerated_failed_amortized_computations: int = 3
    factor_matrix_dtype: torch.dtype = torch.float32
    inverse_exponent_override: dict[int, dict[int, float] | float] = field(default_factory=dict)
```

---

## 4. Distributor Pattern

**Base File**: `/Users/gunale/works/claude_code/optimziation/distributedshampoo/distributed_shampoo/distributor/shampoo_distributor.py`

### Purpose:
- Handles parameter blocking and blocking metadata
- Manages distributed gradient collection and parameter updates
- Supports multiple strategies: DDP, FSDP, etc.

### Key Attributes:

```python
local_blocked_params: list[Tensor]  # Blocks this rank is responsible for
local_block_info_list: list[BlockInfo]  # Metadata for each block
local_masked_blocked_params: list[Tensor]  # Non-None blocks
local_grad_selector: tuple[bool, ...]  # Which blocks have gradients
```

### Key Methods:

```python
def merge_and_block_gradients() -> tuple[Tensor, ...]:
    # Collect and block gradients, return with None selector

def update_params(masked_blocked_search_directions: tuple[Tensor, ...]) -> None:
    # Apply updates to parameters
```

---

## 5. State Management Pattern

### Hierarchical Structure:

```
self.state: dict[Tensor, dict[int, dict[str, Any]]]
  ├─ param1
  │  ├─ block_index_1: dict[str, Any]  # Block-level state
  │  │  ├─ STEP: Tensor (shared across group)
  │  │  ├─ MOMENTUM: Tensor (if momentum > 0)
  │  │  ├─ FILTERED_GRAD: Tensor (if beta1 > 0)
  │  │  └─ SHAMPOO: KroneckerFactorsState (e.g., RootInvShampooKroneckerFactorsState)
  │  │     ├─ factor_matrices: tuple[Tensor, ...]
  │  │     ├─ inv_factor_matrices: tuple[Tensor, ...]
  │  │     └─ factor_matrix_indices: tuple[str, ...]
  │  └─ block_index_2: {...}
  └─ param2: {...}

self._per_group_state_lists: list[dict[str, Any]]  # One per param group
  ├─ [0]: dict[str, Any]
  │  ├─ DISTRIBUTOR: DistributorInterface
  │  ├─ STEP: Tensor (scalar step counter)
  │  ├─ MASKED_BLOCKED_PARAMS: tuple[Tensor, ...]
  │  ├─ MASKED_BLOCKED_GRADS: tuple[Tensor, ...] (None between steps)
  │  ├─ SHAMPOO_PRECONDITIONER_LIST: PreconditionerList
  │  ├─ GRAFTING_PRECONDITIONER_LIST: PreconditionerList (optional)
  │  ├─ MOMENTUM_LIST: tuple[Tensor, ...] (if momentum > 0)
  │  ├─ MASKED_MOMENTUM_LIST: tuple[Tensor, ...]
  │  ├─ FILTERED_GRAD_LIST: tuple[Tensor, ...] (if beta1 > 0)
  │  ├─ MASKED_FILTERED_GRAD_LIST: tuple[Tensor, ...]
  │  └─ PREVIOUS_GRAD_SELECTOR: tuple[bool, ...] or None
  └─ [1]: {...}
```

### Key Constants (shampoo_types.py):

```python
# Optimizer state keys
FILTERED_GRAD = "filtered_grad"
MOMENTUM = "momentum"
STEP = "step"

# Parameter group keys
BETAS = "betas"
BETA3 = "beta3"
EPSILON = "epsilon"
WEIGHT_DECAY = "weight_decay"
MOMENTUM = "momentum"
DAMPENING = "dampening"
LR = "lr"
MAX_PRECONDITIONER_DIM = "max_preconditioner_dim"
PRECONDITION_FREQUENCY = "precondition_frequency"
START_PRECONDITIONING_STEP = "start_preconditioning_step"
USE_BIAS_CORRECTION = "use_bias_correction"
USE_DECOUPLED_WEIGHT_DECAY = "use_decoupled_weight_decay"
USE_NESTEROV = "use_nesterov"
PRECONDITIONER_CONFIG = "preconditioner_config"
GRAFTING_CONFIG = "grafting_config"

# Per-group state keys
DISTRIBUTOR = "distributor"
SHAMPOO_PRECONDITIONER_LIST = "shampoo_preconditioner_list"
GRAFTING_PRECONDITIONER_LIST = "grafting_preconditioner_list"
MOMENTUM_LIST = "momentum_list"
FILTERED_GRAD_LIST = "filtered_grad_list"
MASKED_BLOCKED_PARAMS = "masked_blocked_params"
MASKED_BLOCKED_GRADS = "masked_blocked_grads"
MASKED_MOMENTUM_LIST = "masked_momentum_list"
MASKED_FILTERED_GRAD_LIST = "masked_filtered_grad_list"
```

---

## 6. Integration Point: Preconditioner Config to List

**Method**: `DistributedShampoo._preconditioner_config_to_list_cls()`

Maps configuration classes to preconditioner list implementations:

```python
match preconditioner_config:
    case SGDPreconditionerConfig():
        return SGDPreconditionerList(block_list=..., ...)
    case AdaGradPreconditionerConfig() | RMSpropPreconditionerConfig() | AdamPreconditionerConfig():
        return AdagradPreconditionerList(...)
    case RootInvShampooPreconditionerConfig() | EigendecomposedShampooPreconditionerConfig() | EigenvalueCorrectedShampooPreconditionerConfig():
        return {matching preconditioner list}(...)
    case SignDescentPreconditionerConfig():
        return SignDescentPreconditionerList(...)
    case SpectralDescentPreconditionerConfig():
        return SpectralDescentPreconditionerList(...)
```

This is the **KEY INTEGRATION POINT** for adding Katie!

---

## 7. Key Design Patterns

### 1. **Block Iteration Pattern**:
- Parameters are blocked based on `max_preconditioner_dim`
- Each block gets its own preconditioner state
- Blocks are iterated and processed in lists for efficiency

### 2. **Masked Lists Pattern**:
- Lists with None gradients are "masked" (filtered)
- Prevents expensive operations on empty gradients
- Updated when gradient structure changes

### 3. **Amortized Computation**:
- Expensive operations (eigendecomposition, matrix inverse) done periodically
- Controlled by `precondition_frequency` parameter
- Uses bias correction for first-moment estimates

### 4. **Generic Kronecker Factors**:
- Base class for different Kronecker factor representations
- Wrapped version (stored in state) vs Unwrapped version (used in computation)
- Different types for different Shampoo variants

### 5. **Foreach Operators**:
- Extensive use of `torch._foreach_*` for efficiency
- Operations applied in-place on lists of tensors
- Enables PT2 compilation

---

## 8. What to Implement for DistributedKatie

### A. Create Configuration Class (in `shampoo_types.py`):

```python
@dataclass(kw_only=True)
class KatiePreconditionerConfig(ShampooPreconditionerConfig):
    """Configuration for Katie preconditioner computation.
    
    Katie = Kronecker-factored Approximate Curvature Taylor expansion
    Combines:
    - Diagonal preconditioning (like AdaGrad)
    - Kronecker factor preconditioning (like Shampoo)
    - Separate beta2 parameters for each component
    """
    beta2_diagonal: float = 0.999  # For diagonal component
    diagonal_epsilon: float = 1e-10  # For diagonal stability
    kronecker_epsilon: float = 1e-12  # For Kronecker stability
    # Inherits from ShampooPreconditionerConfig:
    # - amortized_computation_config
    # - factor_matrix_dtype
    # - inverse_exponent_override
```

### B. Create Kronecker Factors State Classes (in `shampoo_preconditioner_list.py`):

```python
@dataclass(kw_only=True)
class KatieKroneckerFactorsState(BaseShampooKroneckerFactorsState):
    """Katie Kronecker factors state."""
    diagonal_second_moment: Tensor  # Diagonal component V_t
    inv_factor_matrices: tuple[Tensor, ...]  # Kronecker inv factors
    
@dataclass(kw_only=True)
class KatieKroneckerFactorsUnwrapped(BaseShampooKroneckerFactorsUnwrapped):
    """Katie Kronecker factors for computation."""
    diagonal_second_moment: Tensor
    inv_factor_matrices: tuple[Tensor, ...]
```

### C. Create Preconditioner List Class (in new file `katie_preconditioner_list.py`):

```python
class KatiePreconditionerList(
    BaseShampooPreconditionerList[
        KatieKroneckerFactorsState,
        KatieKroneckerFactorsUnwrapped,
    ]
):
    """Katie preconditioners combining diagonal + Kronecker factors."""
    
    def __init__(
        self,
        block_list: tuple[Tensor, ...],
        state: Mapping[Tensor, _StateValueType],
        block_info_list: tuple[BlockInfo, ...],
        preconditioner_config: KatiePreconditionerConfig,
        beta2_diagonal: float = 0.999,
        beta2_kronecker: float = 0.999,
        weighting_factor: float = 0.001,  # Usually 1 - beta2
        epsilon: float = 1e-12,
        use_bias_correction: bool = True,
    ) -> None:
        # Initialize both diagonal and Kronecker components
        # Store beta2_diagonal separately
        self._beta2_diagonal = beta2_diagonal
        self._bias_correction2_diagonal: Tensor = torch.tensor(1.0)
        # Rest inherits from BaseShampooPreconditionerList
    
    def update_preconditioners(
        self,
        masked_grad_list: tuple[Tensor, ...],
        step: Tensor,
        perform_amortized_computation: bool,
    ) -> None:
        # 1. Update diagonal component: V_t = beta2_d * V_{t-1} + (1-beta2_d) * g_t^2
        # 2. Update Kronecker components: L, R += weighted outer products
        # 3. Periodically compute Kronecker inverses
    
    def _compute_preconditioned_gradient(
        self,
        grad: Tensor,
        preconditioned_dims_selector: tuple[bool, ...],
        kronecker_factors: KatieKroneckerFactorsUnwrapped,
    ) -> Tensor:
        # Combine: 1. Diagonal preconditioning: g / (sqrt(V) + eps_d)
        #          2. Kronecker preconditioning: L_inv @ result @ R_inv
        #          3. Result scaled by sqrt(diag/kronecker_trace_ratio)?
```

### D. Add Integration in DistributedShampoo:

In `_preconditioner_config_to_list_cls()`:
```python
case KatiePreconditionerConfig():
    return KatiePreconditionerList(
        block_list=state_lists[DISTRIBUTOR].local_blocked_params,
        preconditioner_config=preconditioner_config,
        state=self.state,
        block_info_list=state_lists[DISTRIBUTOR].local_block_info_list,
        beta2_diagonal=preconditioner_config.beta2_diagonal,
        beta2_kronecker=group[BETAS][1],
        weighting_factor=1.0 if group[BETAS][1] == 1.0 else 1 - group[BETAS][1],
        epsilon=group[EPSILON],
        use_bias_correction=group[USE_BIAS_CORRECTION],
    )
```

In `__init__.py`:
```python
from distributed_shampoo.shampoo_types import KatiePreconditionerConfig
__all__ = [..., "KatiePreconditionerConfig"]
```

---

## 9. Testing Considerations

### Test Files to Create:
1. `katie_preconditioner_list_test.py` - Unit tests for preconditioner
2. Integration tests with `DistributedShampoo`

### Key Test Cases:
- Gradient shape handling (1D, 2D, 3D tensors)
- State initialization and blocking
- Amortized computation at correct intervals
- Convergence on simple optimization problems
- Bias correction application
- Compatibility with grafting
- Checkpoint/restore cycle

---

## 10. Implementation Checklist

- [ ] Add `KatiePreconditionerConfig` to `shampoo_types.py`
- [ ] Create `katie_preconditioner_list.py` with state classes and list class
- [ ] Update `_preconditioner_config_to_list_cls()` in `distributed_shampoo.py`
- [ ] Add exports to `__init__.py`
- [ ] Create comprehensive unit tests
- [ ] Add integration tests with DistributedShampoo
- [ ] Test with DDP/FSDP if needed
- [ ] Add documentation

---

## 11. References

### Key Files:
- Main optimizer: `distributed_shampoo/distributed_shampoo.py` (1497 lines)
- Shampoo preconditioner: `distributed_shampoo/preconditioner/shampoo_preconditioner_list.py` (1709 lines)
- Configuration types: `distributed_shampoo/shampoo_types.py` (816 lines)
- Preconditioner base: `distributed_shampoo/preconditioner/preconditioner_list.py` (103 lines)
- Adagrad reference: `distributed_shampoo/preconditioner/adagrad_preconditioner_list.py` (170 lines)

### Pattern Examples:
- **Simple preconditioner**: AdagradPreconditionerList (diagonal only)
- **Complex Kronecker preconditioner**: RootInvShampooPreconditionerList
- **Hybrid approach**: EigenvalueCorrectedShampooPreconditionerList (eigenvectors + corrected eigenvalues)

