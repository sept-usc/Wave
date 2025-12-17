from z3 import Int, And, Or, Solver, ArithRef, Implies, sat, If, BoolRef
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum


class PrecisionType(Enum):
    """Enum for different precision types."""

    HALF = 2  # FP16
    FLOAT = 4  # FP32
    DOUBLE = 8  # FP64


class OperationType(Enum):
    MATMUL = "matmul"
    ADD = "add"


@dataclass
class AlgorithmSpec:
    """Specification of the target algorithm's matrix operations."""

    shapes: List[
        Union[
            Tuple[
                Tuple[str, str, str], Tuple[str, str]
            ],  # for matmul: ((b, m, k), (k, n))
            Tuple[str, str, str],  # for add: (b, s, d)
        ]
    ]
    num_ops: int
    symbols: List[str]  # List of symbolic variable names used in shapes
    granularity: int = 256  # Minimum block size for matrix operations
    max_model_size: Optional[int] = None  # Maximum allowed model size
    min_model_size: Optional[int] = None  # Minimum allowed model size
    gates_num_in_mlp: int = 2
    op_types: Optional[List[OperationType]] = (
        None  # Optional: List of operation types for each op
    )
    op_to_stage: Optional[List[int]] = (
        None  # Optional: maps operation indices to stage indices
    )
    symbol_constraints: Optional[List[BoolRef]] = None


@dataclass
class PMCData:
    """Profiled PMC (Performance Monitoring Counter) data for kernels."""

    flops_half: List[int]  # FLOPS for FP16
    flops_float: List[int]  # FLOPS for FP32
    flops_double: List[int]  # FLOPS for FP64
    gl_load: List[int]
    gl_write: List[int]
    kernel_op_types: Optional[List[OperationType]] = (
        None  # List of operation types for each kernel
    )
    # Stage-based constraints: maps kernel indices to stage indices
    # If None, no stage constraints are applied (backward compatibility)
    kernel_to_stage: Optional[List[int]] = (
        None  # kernel_to_stage[i] = stage index for kernel i
    )

    def __post_init__(self):
        """Validate that all PMC data arrays have the same length."""
        lengths = [
            len(self.flops_half),
            len(self.flops_float),
            len(self.flops_double),
            len(self.gl_load),
            len(self.gl_write),
        ]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError(
                f"All PMC data arrays must have the same length. Got lengths: {lengths}"
            )
        if self.kernel_op_types is not None and len(self.kernel_op_types) != lengths[0]:
            raise ValueError(
                f"kernel_op_types length ({len(self.kernel_op_types)}) must match number of kernels ({lengths[0]})"
            )
        # Validate stage constraints if provided
        if self.kernel_to_stage is not None and len(self.kernel_to_stage) != lengths[0]:
            raise ValueError(
                f"kernel_to_stage length ({len(self.kernel_to_stage)}) must match "
                f"number of kernels ({lengths[0]})"
            )


class WaveVerifier:
    def __init__(
        self,
        algorithm_spec: AlgorithmSpec,
        pmc_data: PMCData,
        add_pmc_upperbound_num: int = 21,
        add_pmc_upperbound_den: int = 20,
        matmul_pmc_upperbound_num: int = 3,
        matmul_pmc_upperbound_den: int = 2,
    ):
        self.algorithm_spec = algorithm_spec
        self.pmc_data = pmc_data
        self.num_kernels = len(pmc_data.flops_half)  # Derive from PMC data
        self.solver = Solver()
        # PMC upperbound ratio for matmul kernels, 4/3 = 1.33 as default
        self.matmul_pmc_upperbound_num = matmul_pmc_upperbound_num
        self.matmul_pmc_upperbound_den = matmul_pmc_upperbound_den
        # PMC upperbound ratio for add kernels, 6/5 = 1.2 as default
        self.add_pmc_upperbound_num = add_pmc_upperbound_num
        self.add_pmc_upperbound_den = add_pmc_upperbound_den

        # Initialize symbolic variables
        self._init_symbols()
        self._init_kernel_variables()
        self._add_constraints()

    def _init_symbols(self):
        """Initialize symbolic variables and constraints for algorithm parameters using symbols from spec."""
        self.symbols = {name: Int(name) for name in self.algorithm_spec.symbols}
        if self.algorithm_spec.symbol_constraints is not None:
            for constraint in self.algorithm_spec.symbol_constraints:
                self.solver.add(constraint)

    def _init_kernel_variables(self):
        """Initialize variables for kernel shapes and mappings."""
        # Kernel shape variables
        self.Ab = [Int(f"Ab{i}") for i in range(self.num_kernels)]
        self.Ax = [Int(f"Ax{i}") for i in range(self.num_kernels)]
        self.Ay = [Int(f"Ay{i}") for i in range(self.num_kernels)]
        self.By = [Int(f"By{i}") for i in range(self.num_kernels)]

        # Kernel to operation mapping
        self.kernel_to_op = [Int(f"kmap{i}") for i in range(self.num_kernels)]

        # Operation start and end indices
        self.op_start = [
            Int(f"start_op{k}") for k in range(self.algorithm_spec.num_ops)
        ]
        self.op_end = [Int(f"end_op{k}") for k in range(self.algorithm_spec.num_ops)]

    def _get_shape_param(self, param: str) -> Union[ArithRef, int]:
        """Helper function to get shape parameter value, handling both symbolic and constant values.

        Args:
            param: Shape parameter name or constant value

        Returns:
            Either a Z3 symbolic variable or an integer constant
        """
        return self.symbols[param] if param in self.symbols else int(param)

    def _get_precision_width(self, i: int) -> int:
        """Get the precision width for a given kernel index.

        The width is determined by comparing FLOPS across different precisions.

        Args:
            i: Index of the kernel to get precision width for

        Returns:
            The precision width in bytes (2 for FP16, 4 for FP32, 8 for FP64)
        """
        # For concrete values, determine width based on which precision has highest FLOPS
        flops = [
            (self.pmc_data.flops_half[i], 2),  # (FLOPS, width)
            (self.pmc_data.flops_float[i], 4),
            (self.pmc_data.flops_double[i], 8),
        ]
        return max(flops, key=lambda x: x[0])[1]

    def _merge_matmul_kernels_for_op(self, k: int):
        """
        Merge a sequence of kernels to form a complete matrix multiplication operation.
        For operation k, this method enforces that the kernels in range [op_start[k], op_end[k]]
        can be combined to perform the matrix multiplication (B x M x K) @ (K x N) = (B x M x N).
        The kernels can be merged in different ways (1, 2, 4, or 8 kernels) with different
        partitioning strategies (e.g., splitting along rows, columns, or both).

        Parameters:
            k -- index of the operation to merge kernels for
        """
        (m1, m2) = self.algorithm_spec.shapes[k]
        # Convert string shape names to their corresponding symbolic variables
        B = self._get_shape_param(m1[0])
        M = self._get_shape_param(m1[1])
        K = self._get_shape_param(m1[2])
        N = self._get_shape_param(m2[1])
        assert m1[2] == m2[0]  # Ensure K dimension matches

        for i in range(self.num_kernels):
            for merge_len in [1, 2]:
                # for merge_len in [1, 2, 4, 8, 16]:
                if i + merge_len > self.num_kernels:
                    continue

                conditions = []
                if merge_len == 1:
                    conditions = [
                        self.Ab[i] == B,
                        self.Ax[i] == M,
                        self.Ay[i] == K,
                        self.By[i] == N,
                    ]
                elif merge_len == 2:
                    conditions = [
                        # Case 1: [A1; A2] @ B, Ab = batch size
                        And(
                            self.Ab[i] == self.Ab[i + 1],
                            self.Ay[i] == self.Ay[i + 1],
                            self.By[i] == self.By[i + 1],
                            self.Ab[i] == B,
                            self.Ax[i] + self.Ax[i + 1] == M,
                            self.Ay[i] == K,
                            self.By[i] == N,
                        ),
                        # Case 2: A @ [B1 | B2], Ab = batch size
                        And(
                            self.Ab[i] == self.Ab[i + 1],
                            self.Ax[i] == self.Ax[i + 1],
                            self.Ay[i] == self.Ay[i + 1],
                            self.Ab[i] == B,
                            self.Ax[i] == M,
                            self.Ay[i] == K,
                            self.By[i] + self.By[i + 1] == N,
                        ),
                        # Case 3: [A1 | A2] @ [B1; B2], Ab = batch size
                        And(
                            self.Ab[i] == self.Ab[i + 1],
                            self.Ax[i] == self.Ax[i + 1],
                            self.By[i] == self.By[i + 1],
                            self.Ab[i] == B,
                            self.Ax[i] == M,
                            self.Ay[i] + self.Ay[i + 1] == K,
                            self.By[i] == N,
                        ),
                        # Case 4: A @ B, Ab1 + Ab2 = B
                        And(
                            self.Ax[i] == self.Ax[i + 1],
                            self.Ay[i] == self.Ay[i + 1],
                            self.By[i] == self.By[i + 1],
                            self.Ab[i] + self.Ab[i + 1] == B,
                            self.Ax[i] == M,
                            self.Ay[i] == K,
                            self.By[i] == N,
                        ),
                    ]
                    conditions = [Or(*conditions)]
                elif merge_len == 4:
                    conditions = [
                        # Case 1: Ab = batch size
                        # Case 1.1
                        # A = [ [A11 | A12]; [A21 | A22] ]
                        # B = [B1; B2]
                        And(
                            self.Ab[i]
                            == self.Ab[i + 1]
                            == self.Ab[i + 2]
                            == self.Ab[i + 3],
                            self.Ax[i] == self.Ax[i + 1],
                            self.Ax[i + 2] == self.Ax[i + 3],
                            self.Ay[i] == self.Ay[i + 2],
                            self.Ay[i + 1] == self.Ay[i + 3],
                            self.By[i]
                            == self.By[i + 1]
                            == self.By[i + 2]
                            == self.By[i + 3],
                            self.Ab[i] == B,
                            self.Ax[i] + self.Ax[i + 2] == M,
                            self.Ay[i] + self.Ay[i + 1] == K,
                            self.By[i] == N,
                        ),
                        # Case 1.2
                        # A = [A1 | A2]
                        # B = [ [B11 | B12]; [B21 | B22] ]
                        And(
                            self.Ab[i]
                            == self.Ab[i + 1]
                            == self.Ab[i + 2]
                            == self.Ab[i + 3],
                            self.Ax[i]
                            == self.Ax[i + 1]
                            == self.Ax[i + 2]
                            == self.Ax[i + 3],
                            self.Ay[i] == self.Ay[i + 2],
                            self.Ay[i + 1] == self.Ay[i + 3],
                            self.By[i] == self.By[i + 1],
                            self.By[i + 2] == self.By[i + 3],
                            self.Ab[i] == B,
                            self.Ax[i] == M,
                            self.Ay[i] + self.Ay[i + 1] == K,
                            self.By[i] + self.By[i + 2] == N,
                        ),
                        # Case 2: Ab splitted
                        # Case 2.1: [A1; A2] @ B
                        And(
                            self.Ab[i] == self.Ab[i + 1],
                            self.Ab[i + 2] == self.Ab[i + 3],
                            self.Ax[i] == self.Ax[i + 2],
                            self.Ax[i + 1] == self.Ax[i + 3],
                            self.Ay[i]
                            == self.Ay[i + 1]
                            == self.Ay[i + 2]
                            == self.Ay[i + 3],
                            self.By[i]
                            == self.By[i + 1]
                            == self.By[i + 2]
                            == self.By[i + 3],
                            self.Ab[i] + self.Ab[i + 2] == B,
                            self.Ax[i] + self.Ax[i + 1] == M,
                            self.Ay[i] == K,
                            self.By[i] == N,
                        ),
                        # Case 2.2: A @ [B1 | B2]
                        And(
                            self.Ab[i] == self.Ab[i + 1],
                            self.Ab[i + 2] == self.Ab[i + 3],
                            self.Ax[i]
                            == self.Ax[i + 1]
                            == self.Ax[i + 2]
                            == self.Ax[i + 3],
                            self.Ay[i]
                            == self.Ay[i + 1]
                            == self.Ay[i + 2]
                            == self.Ay[i + 3],
                            self.By[i] == self.By[i + 2],
                            self.By[i + 1] == self.By[i + 3],
                            self.Ab[i] + self.Ab[i + 2] == B,
                            self.Ax[i] == M,
                            self.Ay[i] == K,
                            self.By[i] + self.By[i + 1] == N,
                        ),
                        # Case 2.3: [A1 | A2] @ [B1; B2]
                        And(
                            self.Ab[i] == self.Ab[i + 2],
                            self.Ab[i + 1] == self.Ab[i + 3],
                            self.Ax[i]
                            == self.Ax[i + 1]
                            == self.Ax[i + 2]
                            == self.Ax[i + 3],
                            self.Ay[i] == self.Ay[i + 2],
                            self.Ay[i + 1] == self.Ay[i + 3],
                            self.By[i]
                            == self.By[i + 1]
                            == self.By[i + 2]
                            == self.By[i + 3],
                            self.Ab[i] + self.Ab[i + 2] == B,
                            self.Ax[i] == M,
                            self.Ay[i] + self.Ay[i + 1] == K,
                            self.By[i] == N,
                        ),
                    ]
                    conditions = [Or(*conditions)]
                elif merge_len == 8:
                    conditions = [
                        # Case 1: Ab = batch size
                        # A = [ [A11 | A12]; [A21 | A22] ]
                        # B = [ [B11 | B12]; [B21 | B22] ]
                        And(
                            self.Ab[i]
                            == self.Ab[i + 1]
                            == self.Ab[i + 2]
                            == self.Ab[i + 3]
                            == self.Ab[i + 4]
                            == self.Ab[i + 5]
                            == self.Ab[i + 6]
                            == self.Ab[i + 7],
                            self.Ax[i]
                            == self.Ax[i + 1]
                            == self.Ax[i + 2]
                            == self.Ax[i + 3],
                            self.Ax[i + 4]
                            == self.Ax[i + 5]
                            == self.Ax[i + 6]
                            == self.Ax[i + 7],
                            self.Ay[i]
                            == self.Ay[i + 2]
                            == self.Ay[i + 4]
                            == self.Ay[i + 6],
                            self.Ay[i + 1]
                            == self.Ay[i + 3]
                            == self.Ay[i + 5]
                            == self.Ay[i + 7],
                            self.By[i]
                            == self.By[i + 1]
                            == self.By[i + 4]
                            == self.By[i + 5],
                            self.By[i + 2]
                            == self.By[i + 3]
                            == self.By[i + 6]
                            == self.By[i + 7],
                            self.Ab[i] == B,
                            self.Ax[i] + self.Ax[i + 4] == M,
                            self.Ay[i] + self.Ay[i + 1] == K,
                            self.By[i] + self.By[i + 2] == N,
                        ),
                        # Case 2: Ab splitted
                        # Case 2.1
                        # A = [ [A11 | A12]; [A21 | A22] ]
                        # B = [B1; B2]
                        And(
                            self.Ab[i]
                            == self.Ab[i + 1]
                            == self.Ab[i + 2]
                            == self.Ab[i + 3],
                            self.Ab[i + 4]
                            == self.Ab[i + 5]
                            == self.Ab[i + 6]
                            == self.Ab[i + 7],
                            self.Ax[i]
                            == self.Ax[i + 1]
                            == self.Ax[i + 4]
                            == self.Ax[i + 5],
                            self.Ax[i + 2]
                            == self.Ax[i + 3]
                            == self.Ax[i + 6]
                            == self.Ax[i + 7],
                            self.Ay[i]
                            == self.Ay[i + 2]
                            == self.Ay[i + 4]
                            == self.Ay[i + 6],
                            self.Ay[i + 1]
                            == self.Ay[i + 3]
                            == self.Ay[i + 5]
                            == self.Ay[i + 7],
                            self.By[i]
                            == self.By[i + 1]
                            == self.By[i + 2]
                            == self.By[i + 3]
                            == self.By[i + 4]
                            == self.By[i + 5]
                            == self.By[i + 6]
                            == self.By[i + 7],
                            self.Ab[i] + self.Ab[i + 4] == B,
                            self.Ax[i] + self.Ax[i + 2] == M,
                            self.Ay[i] + self.Ay[i + 1] == K,
                            self.By[i] == N,
                        ),
                        # Case 2.2
                        # A = [A1 | A2]
                        # B = [ [B11 | B12]; [B21 | B22] ]
                        And(
                            self.Ab[i]
                            == self.Ab[i + 1]
                            == self.Ab[i + 2]
                            == self.Ab[i + 3],
                            self.Ab[i + 4]
                            == self.Ab[i + 5]
                            == self.Ab[i + 6]
                            == self.Ab[i + 7],
                            self.Ax[i]
                            == self.Ax[i + 1]
                            == self.Ax[i + 2]
                            == self.Ax[i + 3]
                            == self.Ax[i + 4]
                            == self.Ax[i + 5]
                            == self.Ax[i + 6]
                            == self.Ax[i + 7],
                            self.Ay[i]
                            == self.Ay[i + 2]
                            == self.Ay[i + 4]
                            == self.Ay[i + 6],
                            self.Ay[i + 1]
                            == self.Ay[i + 3]
                            == self.Ay[i + 5]
                            == self.Ay[i + 7],
                            self.By[i]
                            == self.By[i + 1]
                            == self.By[i + 4]
                            == self.By[i + 5],
                            self.By[i + 2]
                            == self.By[i + 3]
                            == self.By[i + 6]
                            == self.By[i + 7],
                            self.Ab[i] + self.Ab[i + 4] == B,
                            self.Ax[i] == M,
                            self.Ay[i] + self.Ay[i + 1] == K,
                            self.By[i] + self.By[i + 2] == N,
                        ),
                    ]
                    conditions = [Or(*conditions)]
                elif merge_len == 16:
                    # Ab splited
                    # A = [ [A11 | A12]; [A21 | A22] ]
                    # B = [ [B11 | B12]; [B21 | B22] ]
                    conditions = [
                        self.Ab[i]
                        == self.Ab[i + 1]
                        == self.Ab[i + 2]
                        == self.Ab[i + 3]
                        == self.Ab[i + 4]
                        == self.Ab[i + 5]
                        == self.Ab[i + 6]
                        == self.Ab[i + 7],
                        self.Ab[i + 8]
                        == self.Ab[i + 9]
                        == self.Ab[i + 10]
                        == self.Ab[i + 11]
                        == self.Ab[i + 12]
                        == self.Ab[i + 13]
                        == self.Ab[i + 14]
                        == self.Ab[i + 15],
                        self.Ax[i]
                        == self.Ax[i + 1]
                        == self.Ax[i + 2]
                        == self.Ax[i + 3]
                        == self.Ax[i + 8]
                        == self.Ax[i + 9]
                        == self.Ax[i + 10]
                        == self.Ax[i + 11],
                        self.Ax[i + 4]
                        == self.Ax[i + 5]
                        == self.Ax[i + 6]
                        == self.Ax[i + 7]
                        == self.Ax[i + 12]
                        == self.Ax[i + 13]
                        == self.Ax[i + 14]
                        == self.Ax[i + 15],
                        self.Ay[i]
                        == self.Ay[i + 2]
                        == self.Ay[i + 4]
                        == self.Ay[i + 6]
                        == self.Ay[i + 8]
                        == self.Ay[i + 10]
                        == self.Ay[i + 12]
                        == self.Ay[i + 14],
                        self.Ay[i + 1]
                        == self.Ay[i + 3]
                        == self.Ay[i + 5]
                        == self.Ay[i + 7]
                        == self.Ay[i + 9]
                        == self.Ay[i + 11]
                        == self.Ay[i + 13],
                        self.By[i]
                        == self.By[i + 1]
                        == self.By[i + 4]
                        == self.By[i + 5]
                        == self.By[i + 8]
                        == self.By[i + 9]
                        == self.By[i + 12]
                        == self.By[i + 13],
                        self.By[i + 2]
                        == self.By[i + 3]
                        == self.By[i + 6]
                        == self.By[i + 7]
                        == self.By[i + 10]
                        == self.By[i + 11]
                        == self.By[i + 14]
                        == self.By[i + 15],
                        self.Ab[i] + self.Ab[i + 8] == B,
                        self.Ax[i] + self.Ax[i + 4] == M,
                        self.Ay[i] + self.Ay[i + 1] == K,
                        self.By[i] + self.By[i + 2] == N,
                    ]

                self.solver.add(
                    Implies(
                        And(self.op_start[k] == i, self.op_end[k] == i + merge_len - 1),
                        And(*conditions),
                    )
                )

    def _constrain_add_kernel_for_op(self, k: int):
        """
        Constrain a kernel to match the shape of an elementwise addition operation.
        For operation k, this method enforces that the kernel in [op_start[k], op_end[k]]
        matches the shape (B, M, N) for the add operation, and sets By to 1.
        """
        m = self.algorithm_spec.shapes[k]
        B = self._get_shape_param(m[0])
        M = self._get_shape_param(m[1])
        N = self._get_shape_param(m[2])
        for i in range(self.num_kernels):
            self.solver.add(
                Implies(
                    And(self.op_start[k] == i, self.op_end[k] == i),
                    And(
                        self.Ab[i] == B,
                        self.Ax[i] == M,
                        self.Ay[i] == N,
                        self.By[i] == 1,
                    ),
                )
            )

    def _add_constraints(self):
        """Add all constraints to the solver."""
        # Constraint 1: kernel_to_op must be in valid range
        for v in self.kernel_to_op:
            self.solver.add(v >= 0, v < self.algorithm_spec.num_ops)

        # Constraint 2: continuous segments per op
        for k in range(self.algorithm_spec.num_ops):
            self.solver.add(self.op_start[k] >= 0, self.op_start[k] < self.num_kernels)
            self.solver.add(self.op_end[k] >= 0, self.op_end[k] < self.num_kernels)
            self.solver.add(self.op_start[k] <= self.op_end[k])
            k_len = self.op_end[k] - self.op_start[k] + 1
            self.solver.add(
                Or(
                    k_len == 1,
                    k_len == 2,
                    # k_len == 4,
                    # k_len == 8,
                    # k_len == 16,
                )
            )

            # Each kernel in this range must map to op k
            for i in range(self.num_kernels):
                in_range = And(i >= self.op_start[k], i <= self.op_end[k])
                self.solver.add(Implies(in_range, self.kernel_to_op[i] == k))

        # Constraint 3: kernel shapes must be positive and respect granularity
        granularity = self.algorithm_spec.granularity
        for i in range(self.num_kernels):
            self.solver.add(
                self.Ab[i] > 0,
                self.Ax[i] > 0,
                self.Ay[i] > 0,
                self.By[i] > 0,
                Or(self.Ab[i] < granularity, self.Ab[i] % granularity == 0),
                Or(self.Ax[i] < granularity, self.Ax[i] % granularity == 0),
                Or(self.Ay[i] < granularity, self.Ay[i] % granularity == 0),
                Or(self.By[i] < granularity, self.By[i] % granularity == 0),
            )

        # Constraint 4: theory values match PMC with precision width
        for i in range(self.num_kernels):
            if self.pmc_data.kernel_op_types is not None:
                op_type = self.pmc_data.kernel_op_types[i]
            else:
                raise ValueError(f"Kernel operation type is not set for kernel {i}")

            # Get precision width for this kernel
            width = self._get_precision_width(i)

            # FLOPS constraint - use maximum FLOPS across precisions
            max_flops = If(
                self.pmc_data.flops_half[i] > self.pmc_data.flops_float[i],
                If(
                    self.pmc_data.flops_half[i] > self.pmc_data.flops_double[i],
                    self.pmc_data.flops_half[i],
                    self.pmc_data.flops_double[i],
                ),
                If(
                    self.pmc_data.flops_float[i] > self.pmc_data.flops_double[i],
                    self.pmc_data.flops_float[i],
                    self.pmc_data.flops_double[i],
                ),
            )

            if op_type == OperationType.MATMUL:
                self.solver.add(
                    max_flops >= 2 * self.Ab[i] * self.Ax[i] * self.Ay[i] * self.By[i]
                )

                # Global memory constraints - multiply by precision width
                self.solver.add(
                    self.pmc_data.gl_load[i]
                    >= width
                    * (self.Ab[i] * self.Ax[i] * self.Ay[i] + self.Ay[i] * self.By[i])
                )
                self.solver.add(
                    self.pmc_data.gl_load[i] * self.matmul_pmc_upperbound_den
                    <= self.matmul_pmc_upperbound_num
                    * width
                    * (self.Ab[i] * self.Ax[i] * self.Ay[i] + self.Ay[i] * self.By[i])
                )
                self.solver.add(
                    self.pmc_data.gl_write[i]
                    >= width * self.Ab[i] * self.Ax[i] * self.By[i]
                )
            elif op_type == OperationType.ADD:
                self.solver.add(max_flops >= self.Ab[i] * self.Ax[i] * self.Ay[i])
                # self.solver.add(
                #     max_flops * self.add_pmc_upperbound_den
                #     <= self.add_pmc_upperbound_num
                #     * self.Ab[i]
                #     * self.Ax[i]
                #     * self.Ay[i]
                # )
                self.solver.add(
                    self.pmc_data.gl_load[i]
                    >= 2 * width * self.Ab[i] * self.Ax[i] * self.Ay[i]
                )
                self.solver.add(
                    self.pmc_data.gl_load[i] * self.add_pmc_upperbound_den
                    <= self.add_pmc_upperbound_num
                    * 2
                    * width
                    * self.Ab[i]
                    * self.Ax[i]
                    * self.Ay[i]
                )
                self.solver.add(
                    self.pmc_data.gl_write[i]
                    >= width * self.Ab[i] * self.Ax[i] * self.Ay[i]
                )
                self.solver.add(
                    self.pmc_data.gl_write[i] * self.add_pmc_upperbound_den
                    <= self.add_pmc_upperbound_num
                    * width
                    * self.Ab[i]
                    * self.Ax[i]
                    * self.Ay[i]
                )
            else:
                raise ValueError(f"Unknown kernel operation type: {op_type}")

        # Constraint 5: merge kernels for each operation to match required matrix shapes
        for k in range(self.algorithm_spec.num_ops):
            op_type = (
                self.algorithm_spec.op_types[k]
                if self.algorithm_spec.op_types is not None
                else None
            )
            if op_type == OperationType.MATMUL:
                self._merge_matmul_kernels_for_op(k)
            elif op_type == OperationType.ADD:
                self._constrain_add_kernel_for_op(k)
            else:
                raise ValueError(f"Unknown operation type: {op_type}")

        # Constraint 6: op segments appear in increasing kernel index order
        for k in range(self.algorithm_spec.num_ops - 1):
            self.solver.add(self.op_end[k] < self.op_start[k + 1])

        # Constraint 7: enforce model size limit using symbolic variables
        d, d_ffn = self.symbols["d"], self.symbols["d_ffn"]
        if self.algorithm_spec.max_model_size is not None:
            self.solver.add(
                4 * d * d + self.algorithm_spec.gates_num_in_mlp * d * d_ffn
                > self.algorithm_spec.max_model_size
            )
        if self.algorithm_spec.min_model_size is not None:
            self.solver.add(
                4 * d * d + self.algorithm_spec.gates_num_in_mlp * d * d_ffn
                < self.algorithm_spec.min_model_size
            )
        self.solver.add(d_ffn > d)
        self.solver.add(d_ffn < 6 * d)

        # Constraint 8: symbolic variables must respect granularity
        for var in self.symbols.values():
            self.solver.add(var > 0)
            self.solver.add(Or(var < granularity, var % granularity == 0))

        # Constraint 9: stage-based constraints (if provided)
        if (
            self.algorithm_spec.op_to_stage is not None
            and self.pmc_data.kernel_to_stage is not None
        ):
            # For each kernel, it can only be mapped to operations in the same stage
            for i in range(self.num_kernels):
                kernel_stage = self.pmc_data.kernel_to_stage[i]
                kernel_op_type = (
                    self.pmc_data.kernel_op_types[i]
                    if self.pmc_data.kernel_op_types is not None
                    else None
                )
                valid_ops = []
                for k in range(self.algorithm_spec.num_ops):
                    if self.algorithm_spec.op_to_stage[k] == kernel_stage:
                        # Only allow mapping if op type matches
                        if (
                            self.algorithm_spec.op_types is not None
                            and kernel_op_type is not None
                        ):
                            if self.algorithm_spec.op_types[k] == kernel_op_type:
                                valid_ops.append(k)
                        else:
                            valid_ops.append(k)
                if valid_ops:
                    # Kernel can only be mapped to operations in the same stage
                    self.solver.add(
                        Or(*[self.kernel_to_op[i] == op for op in valid_ops])
                    )
                else:
                    # If no operations in the same stage, this is an error
                    raise ValueError(
                        f"Kernel {i} is in stage {kernel_stage}, but no operations are mapped to this stage and type"
                    )

    def verify(self) -> Optional[Dict]:
        """Run the verification and return the solution if found."""
        if self.solver.check() == sat:
            m = self.solver.model()
            solution = {
                "parameters": {name: m[var] for name, var in self.symbols.items()},
                "kernel_configs": [],
                "op_start": [],
                "op_end": [],
            }

            for i in range(self.num_kernels):
                solution["kernel_configs"].append(
                    {
                        "op": m[self.kernel_to_op[i]],
                        "shape": (
                            m[self.Ab[i]],
                            m[self.Ax[i]],
                            m[self.Ay[i]],
                            m[self.By[i]],
                        ),
                    }
                )

            for i in range(self.algorithm_spec.num_ops):
                solution["op_start"].append(m[self.op_start[i]])
                solution["op_end"].append(m[self.op_end[i]])

            return solution
        return None
