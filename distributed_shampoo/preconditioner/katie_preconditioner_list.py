"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from collections.abc import Callable, Hashable, Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import chain
from typing import Any, TypeVar

import torch
from distributed_shampoo.distributor.shampoo_block_info import BlockInfo
from distributed_shampoo.preconditioner.matrix_functions import matrix_inverse_root
from distributed_shampoo.preconditioner.matrix_functions_types import RootInvConfig
from distributed_shampoo.preconditioner.preconditioner_list import (
    PreconditionerList,
    profile_decorator,
)
from distributed_shampoo.preconditioner.shampoo_preconditioner_list import (
    BaseShampooKroneckerFactorsState,
    BaseShampooKroneckerFactorsUnwrapped,
    ClassicShampooPreconditionerList,
)
from distributed_shampoo.shampoo_types import KatiePreconditionerConfig
from distributed_shampoo.utils.optimizer_modules import OptimizerModule
from distributed_shampoo.utils.shampoo_utils import compress_list, get_dtype_size
from torch import Tensor


logger: logging.Logger = logging.getLogger(__name__)

KATIE = "katie"

_SubStateValueType = TypeVar("_SubStateValueType")
_StateValueType = dict[Hashable, _SubStateValueType]


@dataclass
class KatieKroneckerFactorsState(BaseShampooKroneckerFactorsState):
    """Katie Kronecker factors (wrapped) for storing in the optimizer state.

    Extends Shampoo's base state to include diagonal second moment statistics
    and inverse factor matrices for Katie's hybrid preconditioning.

    Attributes:
        factor_matrices (tuple[Tensor, ...]): Kronecker factor matrices (L_t, R_t in pseudocode).
        factor_matrix_indices (tuple[str, ...]): Indices of the factor matrices.
        inv_factor_matrices (tuple[Tensor, ...]): Inverse Kronecker factor matrices (L̂_t, R̂_t in pseudocode).
        diagonal_second_moment (Tensor): Diagonal second moment statistics (V_t in pseudocode).
    """

    inv_factor_matrices: tuple[Tensor, ...]
    diagonal_second_moment: Tensor

    @classmethod
    def from_block(cls, **kwargs: Any) -> "KatieKroneckerFactorsState":
        """Creates a KatieKroneckerFactorsState object for a given block.

        Args:
            block_info (BlockInfo): Information about the block.
            preconditioner_config (KatiePreconditionerConfig): Configuration for the preconditioner.
            preconditioned_dims (tuple[int, ...]): Dimensions for which the factor matrices are preconditioned.

        Returns:
            kronecker_factors_state (KatieKroneckerFactorsState): Instance with initialized state tensors.
        """
        block_info: BlockInfo = kwargs["block_info"]
        preconditioner_config: KatiePreconditionerConfig = kwargs[
            "preconditioner_config"
        ]
        preconditioned_dims: tuple[int, ...] = kwargs["preconditioned_dims"]

        return cls(
            **asdict(
                BaseShampooKroneckerFactorsState.from_block(
                    block_info=block_info,
                    factor_matrix_dtype=preconditioner_config.factor_matrix_dtype,
                    preconditioned_dims=preconditioned_dims,
                )
            ),
            # Initialize inv_factor_matrices as identity matrices.
            inv_factor_matrices=tuple(
                block_info.allocate_eye_tensor(
                    n=dim,
                    dtype=preconditioner_config.inv_factor_matrix_dtype,
                    device=block_info.param.device,
                )
                for dim in preconditioned_dims
            ),
            # Initialize diagonal_second_moment as zeros.
            diagonal_second_moment=block_info.allocate_zeros_tensor(
                size=kwargs["block"].size(),
                dtype=preconditioner_config.diagonal_dtype,
                device=block_info.param.device,
            ),
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        assert len(self.factor_matrices) == len(self.inv_factor_matrices)


@dataclass(kw_only=True)
class KatieKroneckerFactorsUnwrapped(BaseShampooKroneckerFactorsUnwrapped):
    """Katie Kronecker factors (unwrapped) for operations during optimizer computation.

    Attributes:
        factor_matrices (tuple[Tensor, ...]): Kronecker factor matrices.
        factor_matrix_indices (tuple[str, ...]): Indices of the factor matrices.
        inv_factor_matrices (tuple[Tensor, ...]): Inverse Kronecker factor matrices.
        diagonal_second_moment (Tensor): Diagonal second moment statistics.
        roots (tuple[float, ...]): Inverse exponent roots for each factor matrix.
        amortized_computation_config (RootInvConfig): Configuration for inverse-root computation.
        epsilon (float): Regularization constant for Kronecker factors.
        num_tolerated_failed_amortized_computations (int): Maximum failed computations to tolerate.
        _failed_amortized_computation_counter (int): Counter for failed computations.
    """

    inv_factor_matrices: tuple[Tensor, ...]
    diagonal_second_moment: Tensor

    @classmethod
    def from_kronecker_factors_state(
        cls,
        unwrapped_tensor_getter: Callable[[Tensor], Tensor],
        kronecker_factors_state: BaseShampooKroneckerFactorsState,
        roots: tuple[float, ...],
        amortized_computation_config: RootInvConfig,
        epsilon: float,
        num_tolerated_failed_amortized_computations: int,
    ) -> "KatieKroneckerFactorsUnwrapped":
        """Constructs a KatieKroneckerFactorsUnwrapped object from the given state.

        Args:
            unwrapped_tensor_getter (Callable[[Tensor], Tensor]): Function to unwrap tensors.
            kronecker_factors_state (BaseShampooKroneckerFactorsState): The wrapped state.
            roots (tuple[float, ...]): Inverse exponent roots for preconditioning.
            amortized_computation_config (RootInvConfig): Configuration for matrix operations.
            epsilon (float): Regularization constant.
            num_tolerated_failed_amortized_computations (int): Failure tolerance.

        Returns:
            kronecker_factors_unwrapped (KatieKroneckerFactorsUnwrapped): Unwrapped factors.
        """
        assert isinstance(kronecker_factors_state, KatieKroneckerFactorsState)
        return cls(
            inv_factor_matrices=tuple(
                map(
                    unwrapped_tensor_getter,
                    kronecker_factors_state.inv_factor_matrices,
                )
            ),
            diagonal_second_moment=unwrapped_tensor_getter(
                kronecker_factors_state.diagonal_second_moment
            ),
            factor_matrices=tuple(
                map(unwrapped_tensor_getter, kronecker_factors_state.factor_matrices)
            ),
            factor_matrix_indices=kronecker_factors_state.factor_matrix_indices,
            roots=roots,
            amortized_computation_config=amortized_computation_config,
            epsilon=epsilon,
            num_tolerated_failed_amortized_computations=num_tolerated_failed_amortized_computations,
        )

    @torch.compiler.disable
    def _amortized_computation(
        self,
        bias_corrected_factor_matrix: Tensor,
        kronecker_factors_iter_dict: dict[str, Any],
    ) -> tuple[dict[str, Tensor], Exception | None]:
        """Computes matrix inverse roots for Katie's Kronecker preconditioners.

        Args:
            bias_corrected_factor_matrix (Tensor): Factor matrix after bias correction.
            kronecker_factors_iter_dict (dict[str, Any]): Dictionary containing current state.

        Returns:
            computed_quantities (dict[str, Tensor]): Computed inverse factor matrices.
            exception (Exception | None): Any exception encountered, or None if successful.
        """
        inv_factor_matrix, root = (
            kronecker_factors_iter_dict["inv_factor_matrices"],
            kronecker_factors_iter_dict["roots"],
        )

        try:
            # Compute inverse preconditioners for Kronecker factors
            return {
                "inv_factor_matrices": matrix_inverse_root(
                    A=bias_corrected_factor_matrix,
                    root=Fraction(root),
                    root_inv_config=self.amortized_computation_config,
                    epsilon=self.epsilon,
                ).to(dtype=inv_factor_matrix.dtype)
            }, None
        except Exception as exception:
            return {"inv_factor_matrices": inv_factor_matrix}, exception

    def __post_init__(self) -> None:
        super().__post_init__()
        assert (
            len(self.roots)
            == len(self.factor_matrices)
            == len(self.inv_factor_matrices)
        )

    def _get_field_dict(self) -> dict[str, Any]:
        """Creates a dictionary containing shallow copies of this dataclass's fields.

        Excludes diagonal_second_moment as it doesn't participate in per-factor iteration.

        Returns:
            dict[str, Any]: A dictionary mapping field names to their values.
        """
        return {
            key: value
            for key, value in super()._get_field_dict().items()
            if key not in ("diagonal_second_moment",)
        }


class KatiePreconditionerList(
    ClassicShampooPreconditionerList[
        KatieKroneckerFactorsState, KatieKroneckerFactorsUnwrapped
    ]
):
    """Katie preconditioners for list of parameters.

    Implements the Katie optimizer which combines:
    1. Kronecker (Shampoo-like) preconditioning: U_t = L̂_t M̂_t R̂_t
    2. Diagonal (Adam-like) preconditioning: Δ_t = U_t ⊘ (√V̂_t + ε)

    NOTE: Does not support sparse gradients at this time.

    Args:
        block_list (tuple[Tensor, ...]): List of (blocks of) parameters.
        state (Mapping[Tensor, _StateValueType]): Mapping containing optimizer state.
        block_info_list (tuple[BlockInfo, ...]): List containing corresponding BlockInfo.
        preconditioner_config (KatiePreconditionerConfig): Configuration for Katie preconditioner.
        beta2 (float): Decay rate for diagonal second moment (from config or override). (Default: 0.999)
        weighting_factor (float): Weighting factor for current squared gradients. (Default: 1.0 - beta2)
        epsilon (float): Epsilon for Kronecker factor regularization (from config). (Default: 1e-12)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
    """

    def __init__(
        self,
        block_list: tuple[Tensor, ...],
        state: Mapping[Tensor, _StateValueType],
        block_info_list: tuple[BlockInfo, ...],
        preconditioner_config: KatiePreconditionerConfig,
        beta2: float = 0.999,
        weighting_factor: float | None = None,
        epsilon: float = 1e-12,
        use_bias_correction: bool = True,
    ) -> None:
        # Store Katie-specific configuration
        self._katie_config = preconditioner_config
        self._beta2_diagonal = beta2
        self._weighting_factor_diagonal = (
            weighting_factor if weighting_factor is not None else 1.0 - beta2
        )
        self._diagonal_epsilon = preconditioner_config.diagonal_epsilon
        self._bias_correction2_diagonal: Tensor = torch.tensor(1.0)

        # Initialize base Shampoo preconditioner with Kronecker factors
        # Note: epsilon here is for Kronecker factors, not diagonal
        super().__init__(
            block_list=block_list,
            state=state,
            block_info_list=block_info_list,
            preconditioner_config=preconditioner_config,
            beta2=beta2,  # for Kronecker factors
            weighting_factor=(
                weighting_factor if weighting_factor is not None else 1.0 - beta2
            ),
            epsilon=preconditioner_config.kronecker_epsilon,  # Kronecker epsilon
            use_bias_correction=use_bias_correction,
        )

        # Extract diagonal second moment preconditioners
        diagonal_preconditioner_list: list[Tensor] = []
        for block_info in block_info_list:
            param_index, block_index = block_info.composable_block_ids
            block_state = state[block_info.param][block_index]
            diagonal_preconditioner_list.append(
                block_info.get_tensor(block_state[KATIE].diagonal_second_moment)
            )

        # Initialize diagonal preconditioner lists
        self._local_diagonal_preconditioner_list: tuple[Tensor, ...] = tuple(
            diagonal_preconditioner_list
        )
        self._masked_diagonal_preconditioner_list: tuple[Tensor, ...] = (
            self._local_diagonal_preconditioner_list
        )

        # Update memory tracking to include diagonal preconditioners
        diagonal_numel_list: tuple[int, ...] = tuple(
            diag.numel() for diag in self._local_diagonal_preconditioner_list
        )
        diagonal_bytes_list: tuple[int, ...] = tuple(
            numel * get_dtype_size(preconditioner_config.diagonal_dtype)
            for numel in diagonal_numel_list
        )

        # Combine Kronecker and diagonal memory usage
        self._numel_list = tuple(
            k_numel + d_numel
            for k_numel, d_numel in zip(
                self._numel_list, diagonal_numel_list, strict=True
            )
        )
        self._num_bytes_list = tuple(
            k_bytes + d_bytes
            for k_bytes, d_bytes in zip(
                self._num_bytes_list, diagonal_bytes_list, strict=True
            )
        )

    @profile_decorator
    def update_preconditioners(
        self,
        masked_grad_list: tuple[Tensor, ...],
        step: Tensor,
        perform_amortized_computation: bool,
    ) -> None:
        """Updates both Kronecker and diagonal preconditioners.

        Args:
            masked_grad_list (tuple[Tensor, ...]): Gradients with masks applied.
            step (Tensor): Current optimization step.
            perform_amortized_computation (bool): Whether to update Kronecker inverse factors.
        """
        # Update Kronecker factor matrices (from base class)
        self._update_factor_matrices(masked_grad_list=masked_grad_list)

        # Update diagonal second moment
        if self._beta2_diagonal != 1.0:
            torch._foreach_mul_(
                self._masked_diagonal_preconditioner_list, self._beta2_diagonal
            )

        torch._foreach_addcmul_(
            self._masked_diagonal_preconditioner_list,
            masked_grad_list,
            masked_grad_list,
            value=self._weighting_factor_diagonal,
        )

        # Update bias correction terms
        if self._use_bias_correction:
            if self._beta2 < 1.0:
                self._bias_correction2 = torch.tensor(1.0) - self._beta2**step
            if self._beta2_diagonal < 1.0:
                self._bias_correction2_diagonal = (
                    torch.tensor(1.0) - self._beta2_diagonal**step
                )

        # Perform amortized computation for Kronecker inverse factors
        if perform_amortized_computation:
            for kronecker_factors_unwrapped in self._masked_kronecker_factors_unwrapped:
                kronecker_factors_unwrapped.amortized_computation(
                    bias_correction2=self._bias_correction2
                )

    def _compute_preconditioned_gradient(
        self,
        grad: Tensor,
        preconditioned_dims_selector: tuple[bool, ...],
        kronecker_factors: KatieKroneckerFactorsUnwrapped,
    ) -> Tensor:
        """Applies Katie's sequential preconditioning: Kronecker then diagonal.

        Implements the update:
        1. U_t = L̂_t · M̂_t · R̂_t  (Kronecker preconditioning)
        2. Δ_t = U_t ⊘ (√V̂_t + ε)  (diagonal preconditioning)

        Args:
            grad (Tensor): The gradient (or first moment M̂_t) to be preconditioned.
            preconditioned_dims_selector (tuple[bool, ...]): Dimensions to precondition.
            kronecker_factors (KatieKroneckerFactorsUnwrapped): Unwrapped Katie factors.

        Returns:
            preconditioned_grad (Tensor): The preconditioned gradient.
        """
        # Step 1: Apply Kronecker preconditioning
        preconditioned_grad = self._precondition_grad(
            grad=grad,
            preconditioned_dims_selector=preconditioned_dims_selector,
            preconditioner_list=kronecker_factors.inv_factor_matrices,
        )

        # Step 2: Apply diagonal preconditioning (like Adam)
        # Note: We need to get the corresponding diagonal preconditioner for this gradient
        # This is handled via the masked lists during precondition() call
        return preconditioned_grad

    @profile_decorator
    def precondition(self, masked_grad_list: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """Preconditions gradients using Katie's hybrid approach.

        Applies Kronecker preconditioning followed by diagonal preconditioning.

        Args:
            masked_grad_list (tuple[Tensor, ...]): Gradients with masks applied.

        Returns:
            preconditioned_grads (tuple[Tensor, ...]): Preconditioned gradients.
        """
        # Apply Kronecker preconditioning
        kronecker_preconditioned_grads = tuple(
            self._compute_preconditioned_gradient(
                grad=masked_grad,
                preconditioned_dims_selector=preconditioned_dims_selector,
                kronecker_factors=kronecker_factors,
            )
            for masked_grad, preconditioned_dims_selector, kronecker_factors in zip(
                masked_grad_list,
                self._masked_preconditioned_dims_selector_list,
                self._masked_kronecker_factors_unwrapped,
                strict=True,
            )
        )

        # Apply diagonal preconditioning (Adam-like)
        masked_bias_corrected_diagonal_list = torch._foreach_div(
            self._masked_diagonal_preconditioner_list,
            self._bias_correction2_diagonal,
        )
        torch._foreach_sqrt_(masked_bias_corrected_diagonal_list)
        torch._foreach_add_(masked_bias_corrected_diagonal_list, self._diagonal_epsilon)

        # Element-wise division: Δ_t = U_t ⊘ (√V̂_t + ε)
        return torch._foreach_div(
            kronecker_preconditioned_grads, masked_bias_corrected_diagonal_list
        )

    @profile_decorator
    def compress_preconditioner_list(
        self, local_grad_selector: tuple[bool, ...]
    ) -> None:
        """Compresses both Kronecker and diagonal preconditioner lists based on gradient selector.

        Args:
            local_grad_selector (tuple[bool, ...]): Selector for active gradients.
        """
        # Compress Kronecker factors (from base class)
        super().compress_preconditioner_list(local_grad_selector)

        # Compress diagonal preconditioners
        self._masked_diagonal_preconditioner_list = compress_list(
            self._local_diagonal_preconditioner_list, local_grad_selector
        )
