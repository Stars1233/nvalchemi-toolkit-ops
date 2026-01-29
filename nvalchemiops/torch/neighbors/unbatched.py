# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch bindings for unbatched (single-system) neighbor list construction.

This module provides PyTorch custom operators for naive and cell list methods.
"""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.neighbors.cell_list import (
    build_cell_list as wp_build_cell_list,
)
from nvalchemiops.neighbors.cell_list import (
    query_cell_list as wp_query_cell_list,
)
from nvalchemiops.neighbors.naive import (
    naive_neighbor_matrix,
    naive_neighbor_matrix_pbc,
)
from nvalchemiops.neighbors.naive_dual_cutoff import (
    naive_neighbor_matrix_dual_cutoff,
    naive_neighbor_matrix_pbc_dual_cutoff,
)
from nvalchemiops.neighbors.neighbor_utils import (
    _expand_naive_shifts,
    estimate_max_neighbors,
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    allocate_cell_list,
    compute_naive_num_shifts,
    get_neighbor_list_from_neighbor_matrix,
)
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

__all__ = [
    "naive_neighbor_list",
    "naive_neighbor_list_dual_cutoff",
    "cell_list",
    "build_cell_list",
    "query_cell_list",
    "estimate_cell_list_sizes",
]

###########################################################################################
########################### Naive Neighbor List PyTorch Bindings #########################
###########################################################################################


@torch.library.custom_op(
    "nvalchemiops::_naive_neighbor_matrix_no_pbc",
    mutates_args=("neighbor_matrix", "num_neighbors"),
)
def _naive_neighbor_matrix_no_pbc(
    positions: torch.Tensor,
    cutoff: float,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    half_fill: bool = False,
) -> None:
    """Fill neighbor matrix for atoms using naive O(N^2) algorithm.

    Custom PyTorch operator that computes pairwise distances and fills
    the neighbor matrix with atom indices within the cutoff distance.
    No periodic boundary conditions are applied.

    This function does not allocate any tensors.

    This function is torch compilable.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3), dtype=torch.float32 or torch.float64
        Atomic coordinates in Cartesian space. Each row represents one atom's
        (x, y, z) position.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    neighbor_matrix : torch.Tensor, shape (total_atoms, max_neighbors), dtype=torch.int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
        Must be pre-allocated. Entries are filled with atom indices.
    num_neighbors : torch.Tensor, shape (total_atoms,), dtype=torch.int32
        OUTPUT: Number of neighbors found for each atom.
        Must be pre-allocated. Updated in-place with actual neighbor counts.
    half_fill : bool
        If True, only store relationships where i < j to avoid double counting.
        If False, store all neighbor relationships symmetrically.

    See Also
    --------
    nvalchemiops.neighbors.naive.naive_neighbor_matrix : Core warp launcher
    naive_neighbor_list : High-level wrapper function
    """
    device = positions.device
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)

    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True)

    naive_neighbor_matrix(
        positions=wp_positions,
        cutoff=cutoff,
        neighbor_matrix=wp_neighbor_matrix,
        num_neighbors=wp_num_neighbors,
        wp_dtype=wp_dtype,
        device=str(device),
        half_fill=half_fill,
    )


@torch.library.custom_op(
    "nvalchemiops::_naive_neighbor_matrix_pbc",
    mutates_args=("neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"),
)
def _naive_neighbor_matrix_pbc(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    shift_range_per_dimension: torch.Tensor,
    shift_offset: torch.Tensor,
    total_shifts: int,
    half_fill: bool = False,
) -> None:
    """Compute neighbor matrix with periodic boundary conditions using naive O(N^2) algorithm.

    This function assumes that the number of shifts has been computed and the shifts have been
    expanded into a single array of shift vectors. PBC information is encoded in the pre-computed
    shifts, so it's not needed as a separate argument.

    This function is torch compilable.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3), dtype=torch.float32 or torch.float64
        Atomic coordinates in Cartesian space. Each row represents one atom's
        (x, y, z) position.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    cell : torch.Tensor, shape (1, 3, 3), dtype=torch.float32 or torch.float64
        Cell matrices defining lattice vectors in Cartesian coordinates.
    neighbor_matrix : torch.Tensor, shape (total_atoms, max_neighbors), dtype=torch.int32
        OUTPUT: Neighbor matrix to be filled.
    neighbor_matrix_shifts : torch.Tensor, shape (total_atoms, max_neighbors, 3), dtype=torch.int32
        OUTPUT: Shift vectors for each neighbor relationship.
    num_neighbors : torch.Tensor, shape (total_atoms,), dtype=torch.int32
        OUTPUT: Number of neighbors found for each atom.
    shift_range_per_dimension : torch.Tensor, shape (1, 3), dtype=torch.int32
        Shift range in each dimension for each system.
    shift_offset : torch.Tensor, shape (2,), dtype=torch.int32
        Cumulative sum of number of shifts for each system.
    total_shifts : int
        Total number of shifts.
    half_fill : bool, optional
        If True, only store relationships where i < j to avoid double counting.
        If False, store all neighbor relationships symmetrically. Default is False.

    Notes
    -----
    The PBC parameter is not needed because PBC information is encoded in the
    pre-computed shift vectors (shift_range_per_dimension, shift_offset).

    See Also
    --------
    nvalchemiops.neighbors.naive.naive_neighbor_matrix_pbc : Core warp launcher
    nvalchemiops.neighbors.neighbor_utils._expand_naive_shifts : Kernel for expanding shifts
    naive_neighbor_list : High-level wrapper function
    """
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(cell.dtype)

    # Expand shift ranges into explicit shift vectors
    shifts = torch.empty((total_shifts, 3), dtype=torch.int32, device=device)
    shift_system_idx = torch.empty((total_shifts,), dtype=torch.int32, device=device)
    wp_shifts = wp.from_torch(shifts, dtype=wp.vec3i, return_ctype=True)
    wp_shift_system_idx = wp.from_torch(
        shift_system_idx, dtype=wp.int32, return_ctype=True
    )
    wp_shift_range_per_dimension = wp.from_torch(
        shift_range_per_dimension, dtype=wp.vec3i, return_ctype=True
    )
    wp_shift_offset = wp.from_torch(shift_offset, dtype=wp.int32, return_ctype=True)

    wp.launch(
        kernel=_expand_naive_shifts,
        dim=1,
        inputs=[
            wp_shift_range_per_dimension,
            wp_shift_offset,
            wp_shifts,
            wp_shift_system_idx,
        ],
        device=wp_device,
    )

    # Call warp launcher for neighbor computation
    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, return_ctype=True
    )
    wp_neighbor_matrix_shifts = wp.from_torch(
        neighbor_matrix_shifts, dtype=wp.vec3i, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True)

    naive_neighbor_matrix_pbc(
        positions=wp_positions,
        cutoff=cutoff,
        cell=wp_cell,
        shifts=wp_shifts,
        neighbor_matrix=wp_neighbor_matrix,
        neighbor_matrix_shifts=wp_neighbor_matrix_shifts,
        num_neighbors=wp_num_neighbors,
        wp_dtype=wp_dtype,
        device=str(device),
        half_fill=half_fill,
    )


def naive_neighbor_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
    max_neighbors: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    num_neighbors: torch.Tensor | None = None,
    shift_range_per_dimension: torch.Tensor | None = None,
    shift_offset: torch.Tensor | None = None,
    total_shifts: int | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor]
):
    """Compute neighbor list using naive O(N^2) algorithm.

    Identifies all atom pairs within a specified cutoff distance using a
    brute-force pairwise distance calculation. Supports both non-periodic
    and periodic boundary conditions.

    For non-pbc systems, this function is torch compilable. For pbc systems,
    precompute the shift vectors using compute_naive_num_shifts.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3), dtype=torch.float32 or torch.float64
        Atomic coordinates in Cartesian space. Each row represents one atom's
        (x, y, z) position.
    cutoff : float
        Cutoff distance for neighbor detection in Cartesian units.
        Must be positive. Atoms within this distance are considered neighbors.
    pbc : torch.Tensor, shape (1, 3), dtype=torch.bool, optional
        Periodic boundary condition flags for each dimension.
        True enables periodicity in that direction. Default is None (no PBC).
    cell : torch.Tensor, shape (1, 3, 3), dtype=torch.float32 or torch.float64, optional
        Cell matrices defining lattice vectors in Cartesian coordinates.
        Required if pbc is provided. Default is None.
    max_neighbors : int, optional
        Maximum number of neighbors per atom. Must be positive.
        If exceeded, excess neighbors are ignored.
        Must be provided if neighbor_matrix is not provided.
    half_fill : bool, optional
        If True, only store relationships where i < j to avoid double counting.
        If False, store all neighbor relationships symmetrically. Default is False.
    fill_value : int, optional
        Value to fill the neighbor matrix with. Default is total_atoms.
    neighbor_matrix : torch.Tensor, shape (total_atoms, max_neighbors), dtype=torch.int32, optional
        Neighbor matrix to be filled. Pass in a pre-allocated tensor to avoid reallocation.
        Must be provided if max_neighbors is not provided.
    neighbor_matrix_shifts : torch.Tensor, shape (total_atoms, max_neighbors, 3), dtype=torch.int32, optional
        Shift vectors for each neighbor relationship. Pass in a pre-allocated tensor to avoid reallocation.
        Must be provided if max_neighbors is not provided.
    num_neighbors : torch.Tensor, shape (total_atoms,), dtype=torch.int32, optional
        Number of neighbors found for each atom. Pass in a pre-allocated tensor to avoid reallocation.
        Must be provided if max_neighbors is not provided.
    shift_range_per_dimension : torch.Tensor, shape (1, 3), dtype=torch.int32, optional
        Shift range in each dimension for each system.
        Pass in a pre-allocated tensor to avoid reallocation for pbc systems.
    shift_offset : torch.Tensor, shape (2,), dtype=torch.int32, optional
        Cumulative sum of number of shifts for each system.
        Pass in a pre-allocated tensor to avoid reallocation for pbc systems.
    total_shifts : int, optional
        Total number of shifts.
        Pass in a pre-allocated tensor to avoid reallocation for pbc systems.
    return_neighbor_list : bool, optional - default = False
        If True, convert the neighbor matrix to a neighbor list (idx_i, idx_j) format by
        creating a mask over the fill_value, which can incur a performance penalty.
        We recommend using the neighbor matrix format,
        and only convert to a neighbor list format if absolutely necessary.

    Returns
    -------
    results : tuple of torch.Tensor
        Variable-length tuple depending on input parameters. The return pattern follows:

        - No PBC, matrix format: ``(neighbor_matrix, num_neighbors)``
        - No PBC, list format: ``(neighbor_list, neighbor_ptr)``
        - With PBC, matrix format: ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)``
        - With PBC, list format: ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)``

        **Components returned:**

        - **neighbor_data** (tensor): Neighbor indices, format depends on ``return_neighbor_list``:

            * If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix``
              with shape (total_atoms, max_neighbors), dtype int32. Each row i contains
              indices of atom i's neighbors.
            * If ``return_neighbor_list=True``: Returns ``neighbor_list`` with shape
              (2, num_pairs), dtype int32, in COO format [source_atoms, target_atoms].

        - **num_neighbor_data** (tensor): Information about the number of neighbors for each atom,
          format depends on ``return_neighbor_list``:

            * If ``return_neighbor_list=False`` (default): Returns ``num_neighbors`` with shape (total_atoms,), dtype int32.
              Count of neighbors found for each atom. Always returned.
            * If ``return_neighbor_list=True``: Returns ``neighbor_ptr`` with shape (total_atoms + 1,), dtype int32.
              CSR-style pointer arrays where ``neighbor_ptr_data[i]`` to ``neighbor_ptr_data[i+1]`` gives the range of
              neighbors for atom i in the flattened neighbor list.

        - **neighbor_shift_data** (tensor, optional): Periodic shift vectors, only when ``pbc`` is provided:
          format depends on ``return_neighbor_list``:

            * If ``return_neighbor_list=False`` (default): Returns ``neighbor_matrix_shifts`` with
              shape (total_atoms, max_neighbors, 3), dtype int32.
            * If ``return_neighbor_list=True``: Returns ``unit_shifts`` with shape
              (num_pairs, 3), dtype int32.

    Examples
    --------
    Basic usage without periodic boundary conditions:

    >>> import torch
    >>> positions = torch.rand(100, 3) * 10.0  # 100 atoms in 10x10x10 box
    >>> cutoff = 2.5
    >>> max_neighbors = 50
    >>> neighbor_matrix, num_neighbors = naive_neighbor_list(
    ...     positions, cutoff, max_neighbors
    ... )
    >>> print(f"Found {num_neighbors.sum()} total neighbor pairs")

    With periodic boundary conditions:

    >>> cell = torch.eye(3).unsqueeze(0) * 10.0  # 10x10x10 cubic cell
    >>> pbc = torch.tensor([[True, True, True]])  # Periodic in all directions
    >>> neighbor_matrix, num_neighbors, shifts = naive_neighbor_list(
    ...     positions, cutoff, max_neighbors, pbc=pbc, cell=cell
    ... )

    Return as neighbor list instead of matrix:

    >>> neighbor_list, neighbor_ptr = naive_neighbor_list(
    ...     positions, cutoff, max_neighbors, return_neighbor_list=True
    ... )
    >>> source_atoms, target_atoms = neighbor_list[0], neighbor_list[1]

    See Also
    --------
    nvalchemiops.neighbors.naive.naive_neighbor_matrix : Core warp launcher (no PBC)
    nvalchemiops.neighbors.naive.naive_neighbor_matrix_pbc : Core warp launcher (with PBC)
    cell_list : O(N) cell list method for larger systems
    """
    if pbc is None and cell is not None:
        raise ValueError("If cell is provided, pbc must also be provided")
    if pbc is not None and cell is None:
        raise ValueError("If pbc is provided, cell must also be provided")

    if cell is not None:
        cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
    if pbc is not None:
        pbc = pbc if pbc.ndim == 2 else pbc.unsqueeze(0)

    if max_neighbors is None and (
        neighbor_matrix is None
        or (neighbor_matrix_shifts is None and pbc is not None)
        or num_neighbors is None
    ):
        max_neighbors = estimate_max_neighbors(cutoff)

    if fill_value is None:
        fill_value = positions.shape[0]

    if neighbor_matrix is None:
        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors),
            fill_value,
            dtype=torch.int32,
            device=positions.device,
        )
    else:
        neighbor_matrix.fill_(fill_value)

    if num_neighbors is None:
        num_neighbors = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=positions.device
        )
    else:
        num_neighbors.zero_()

    if pbc is not None:
        if neighbor_matrix_shifts is None:
            neighbor_matrix_shifts = torch.zeros(
                (positions.shape[0], max_neighbors, 3),
                dtype=torch.int32,
                device=positions.device,
            )
        else:
            neighbor_matrix_shifts.zero_()
        if (
            total_shifts is None
            or shift_offset is None
            or shift_range_per_dimension is None
        ):
            shift_range_per_dimension, shift_offset, total_shifts = (
                compute_naive_num_shifts(cell, cutoff, pbc)
            )

    if cutoff <= 0:
        if return_neighbor_list:
            if pbc is not None:
                return (
                    torch.zeros((2, 0), dtype=torch.int32, device=positions.device),
                    torch.zeros(
                        (positions.shape[0] + 1,),
                        dtype=torch.int32,
                        device=positions.device,
                    ),
                    torch.zeros((0, 3), dtype=torch.int32, device=positions.device),
                )
            else:
                return (
                    torch.zeros((2, 0), dtype=torch.int32, device=positions.device),
                    torch.zeros(
                        (positions.shape[0] + 1,),
                        dtype=torch.int32,
                        device=positions.device,
                    ),
                )
        else:
            if pbc is not None:
                return neighbor_matrix, num_neighbors, neighbor_matrix_shifts
            else:
                return neighbor_matrix, num_neighbors

    if pbc is None:
        _naive_neighbor_matrix_no_pbc(
            positions=positions,
            cutoff=cutoff,
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            half_fill=half_fill,
        )
        if return_neighbor_list:
            neighbor_list, neighbor_ptr = get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix,
                num_neighbors=num_neighbors,
                fill_value=fill_value,
            )
            return neighbor_list, neighbor_ptr
        else:
            return neighbor_matrix, num_neighbors
    else:
        _naive_neighbor_matrix_pbc(
            positions=positions,
            cutoff=cutoff,
            cell=cell,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            num_neighbors=num_neighbors,
            shift_range_per_dimension=shift_range_per_dimension,
            shift_offset=shift_offset,
            total_shifts=total_shifts,
            half_fill=half_fill,
        )
        if return_neighbor_list:
            neighbor_list, neighbor_ptr, neighbor_list_shifts = (
                get_neighbor_list_from_neighbor_matrix(
                    neighbor_matrix,
                    num_neighbors=num_neighbors,
                    neighbor_shift_matrix=neighbor_matrix_shifts,
                    fill_value=fill_value,
                )
            )
            return neighbor_list, neighbor_ptr, neighbor_list_shifts
        else:
            return neighbor_matrix, num_neighbors, neighbor_matrix_shifts


###########################################################################################
########################### Naive Dual Cutoff PyTorch Bindings ############################
###########################################################################################


@torch.library.custom_op(
    "nvalchemiops::_naive_neighbor_matrix_no_pbc_dual_cutoff",
    mutates_args=(
        "neighbor_matrix1",
        "num_neighbors1",
        "neighbor_matrix2",
        "num_neighbors2",
    ),
)
def _naive_neighbor_matrix_no_pbc_dual_cutoff(
    positions: torch.Tensor,
    cutoff1: float,
    cutoff2: float,
    neighbor_matrix1: torch.Tensor,
    num_neighbors1: torch.Tensor,
    neighbor_matrix2: torch.Tensor,
    num_neighbors2: torch.Tensor,
    half_fill: bool = False,
) -> None:
    """Fill two neighbor matrices for atoms using dual cutoffs with naive O(N^2) algorithm.

    This function is torch compilable.

    See Also
    --------
    nvalchemiops.neighbors.naive_dual_cutoff.naive_neighbor_matrix_dual_cutoff : Core warp launcher
    naive_neighbor_list_dual_cutoff : High-level wrapper function
    """
    device = positions.device
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_dtype = get_wp_dtype(positions.dtype)

    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_neighbor_matrix1 = wp.from_torch(
        neighbor_matrix1, dtype=wp.int32, return_ctype=True
    )
    wp_num_neighbors1 = wp.from_torch(num_neighbors1, dtype=wp.int32, return_ctype=True)
    wp_neighbor_matrix2 = wp.from_torch(
        neighbor_matrix2, dtype=wp.int32, return_ctype=True
    )
    wp_num_neighbors2 = wp.from_torch(num_neighbors2, dtype=wp.int32, return_ctype=True)

    naive_neighbor_matrix_dual_cutoff(
        positions=wp_positions,
        cutoff1=cutoff1,
        cutoff2=cutoff2,
        neighbor_matrix1=wp_neighbor_matrix1,
        num_neighbors1=wp_num_neighbors1,
        neighbor_matrix2=wp_neighbor_matrix2,
        num_neighbors2=wp_num_neighbors2,
        wp_dtype=wp_dtype,
        device=str(device),
        half_fill=half_fill,
    )


@torch.library.custom_op(
    "nvalchemiops::_naive_neighbor_matrix_pbc_dual_cutoff",
    mutates_args=(
        "neighbor_matrix1",
        "neighbor_matrix2",
        "neighbor_matrix_shifts1",
        "neighbor_matrix_shifts2",
        "num_neighbors1",
        "num_neighbors2",
    ),
)
def _naive_neighbor_matrix_pbc_dual_cutoff(
    positions: torch.Tensor,
    cutoff1: float,
    cutoff2: float,
    cell: torch.Tensor,
    neighbor_matrix1: torch.Tensor,
    neighbor_matrix2: torch.Tensor,
    neighbor_matrix_shifts1: torch.Tensor,
    neighbor_matrix_shifts2: torch.Tensor,
    num_neighbors1: torch.Tensor,
    num_neighbors2: torch.Tensor,
    shift_range_per_dimension: torch.Tensor,
    shift_offset: torch.Tensor,
    total_shifts: int,
    half_fill: bool = False,
) -> None:
    """Compute two neighbor matrices with periodic boundary conditions using dual cutoffs.

    This function is torch compilable.

    See Also
    --------
    nvalchemiops.neighbors.naive_dual_cutoff.naive_neighbor_matrix_pbc_dual_cutoff : Core warp launcher
    naive_neighbor_list_dual_cutoff : High-level wrapper function
    """
    device = positions.device
    wp_device = wp.device_from_torch(device)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    wp_dtype = get_wp_dtype(positions.dtype)

    # Expand shift ranges into explicit shift vectors
    shifts = torch.empty((total_shifts, 3), dtype=torch.int32, device=device)
    shift_system_idx = torch.empty((total_shifts,), dtype=torch.int32, device=device)
    wp_shifts = wp.from_torch(shifts, dtype=wp.vec3i, return_ctype=True)
    wp_shift_system_idx = wp.from_torch(
        shift_system_idx, dtype=wp.int32, return_ctype=True
    )

    wp.launch(
        kernel=_expand_naive_shifts,
        dim=1,
        inputs=[
            wp.from_torch(shift_range_per_dimension, dtype=wp.vec3i, return_ctype=True),
            wp.from_torch(shift_offset, dtype=wp.int32, return_ctype=True),
            wp_shifts,
            wp_shift_system_idx,
        ],
        device=wp_device,
    )

    # Convert tensors to warp arrays
    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_neighbor_matrix1 = wp.from_torch(
        neighbor_matrix1, dtype=wp.int32, return_ctype=True
    )
    wp_neighbor_matrix2 = wp.from_torch(
        neighbor_matrix2, dtype=wp.int32, return_ctype=True
    )
    wp_neighbor_matrix_shifts1 = wp.from_torch(
        neighbor_matrix_shifts1, dtype=wp.vec3i, return_ctype=True
    )
    wp_neighbor_matrix_shifts2 = wp.from_torch(
        neighbor_matrix_shifts2, dtype=wp.vec3i, return_ctype=True
    )
    wp_num_neighbors1 = wp.from_torch(num_neighbors1, dtype=wp.int32, return_ctype=True)
    wp_num_neighbors2 = wp.from_torch(num_neighbors2, dtype=wp.int32, return_ctype=True)

    naive_neighbor_matrix_pbc_dual_cutoff(
        positions=wp_positions,
        cutoff1=cutoff1,
        cutoff2=cutoff2,
        cell=wp_cell,
        shifts=wp_shifts,
        neighbor_matrix1=wp_neighbor_matrix1,
        neighbor_matrix2=wp_neighbor_matrix2,
        neighbor_matrix_shifts1=wp_neighbor_matrix_shifts1,
        neighbor_matrix_shifts2=wp_neighbor_matrix_shifts2,
        num_neighbors1=wp_num_neighbors1,
        num_neighbors2=wp_num_neighbors2,
        wp_dtype=wp_dtype,
        device=str(device),
        half_fill=half_fill,
    )


def naive_neighbor_list_dual_cutoff(
    positions: torch.Tensor,
    cutoff1: float,
    cutoff2: float,
    pbc: torch.Tensor | None = None,
    cell: torch.Tensor | None = None,
    max_neighbors1: int | None = None,
    max_neighbors2: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    neighbor_matrix1: torch.Tensor | None = None,
    neighbor_matrix2: torch.Tensor | None = None,
    neighbor_matrix_shifts1: torch.Tensor | None = None,
    neighbor_matrix_shifts2: torch.Tensor | None = None,
    num_neighbors1: torch.Tensor | None = None,
    num_neighbors2: torch.Tensor | None = None,
    shift_range_per_dimension: torch.Tensor | None = None,
    shift_offset: torch.Tensor | None = None,
    total_shifts: int | None = None,
) -> (
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Compute neighbor list using naive O(N^2) algorithm with dual cutoffs.

    Identifies all atom pairs within two different cutoff distances using a
    single brute-force pairwise distance calculation. This is more efficient
    than running two separate neighbor calculations when both neighbor lists are needed.

    See Also
    --------
    nvalchemiops.neighbors.naive_dual_cutoff.naive_neighbor_matrix_dual_cutoff : Core warp launcher (no PBC)
    nvalchemiops.neighbors.naive_dual_cutoff.naive_neighbor_matrix_pbc_dual_cutoff : Core warp launcher (with PBC)
    naive_neighbor_list : Single cutoff version
    """
    if pbc is None and cell is not None:
        raise ValueError("If cell is provided, pbc must also be provided")
    if pbc is not None and cell is None:
        raise ValueError("If pbc is provided, cell must also be provided")

    if cell is not None:
        cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
    if pbc is not None:
        pbc = pbc if pbc.ndim == 2 else pbc.unsqueeze(0)

    if fill_value is None:
        fill_value = positions.shape[0]

    if max_neighbors1 is None and (
        neighbor_matrix1 is None
        or neighbor_matrix2 is None
        or (neighbor_matrix_shifts1 is None and pbc is not None)
        or (neighbor_matrix_shifts2 is None and pbc is not None)
        or num_neighbors1 is None
        or num_neighbors2 is None
    ):
        max_neighbors2 = estimate_max_neighbors(cutoff2)
        max_neighbors1 = max_neighbors2

    if max_neighbors2 is None:
        max_neighbors2 = max_neighbors1

    if neighbor_matrix1 is None:
        neighbor_matrix1 = torch.full(
            (positions.shape[0], max_neighbors1),
            fill_value,
            dtype=torch.int32,
            device=positions.device,
        )
    else:
        neighbor_matrix1.fill_(fill_value)

    if num_neighbors1 is None:
        num_neighbors1 = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=positions.device
        )
    else:
        num_neighbors1.zero_()

    if neighbor_matrix2 is None:
        neighbor_matrix2 = torch.full(
            (positions.shape[0], max_neighbors2),
            fill_value,
            dtype=torch.int32,
            device=positions.device,
        )
    else:
        neighbor_matrix2.fill_(fill_value)

    if num_neighbors2 is None:
        num_neighbors2 = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=positions.device
        )
    else:
        num_neighbors2.zero_()

    if pbc is not None:
        if neighbor_matrix_shifts1 is None:
            neighbor_matrix_shifts1 = torch.zeros(
                (positions.shape[0], max_neighbors1, 3),
                dtype=torch.int32,
                device=positions.device,
            )
        else:
            neighbor_matrix_shifts1.zero_()
        if neighbor_matrix_shifts2 is None:
            neighbor_matrix_shifts2 = torch.zeros(
                (positions.shape[0], max_neighbors2, 3),
                dtype=torch.int32,
                device=positions.device,
            )
        else:
            neighbor_matrix_shifts2.zero_()
        if (
            total_shifts is None
            or shift_offset is None
            or shift_range_per_dimension is None
        ):
            shift_range_per_dimension, shift_offset, total_shifts = (
                compute_naive_num_shifts(cell, cutoff2, pbc)
            )

    if pbc is None:
        _naive_neighbor_matrix_no_pbc_dual_cutoff(
            positions=positions,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            neighbor_matrix1=neighbor_matrix1,
            num_neighbors1=num_neighbors1,
            neighbor_matrix2=neighbor_matrix2,
            num_neighbors2=num_neighbors2,
            half_fill=half_fill,
        )
        if return_neighbor_list:
            neighbor_list1, neighbor_ptr1 = get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix1, num_neighbors=num_neighbors1, fill_value=fill_value
            )
            neighbor_list2, neighbor_ptr2 = get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix2, num_neighbors=num_neighbors2, fill_value=fill_value
            )
            return (
                neighbor_list1,
                neighbor_ptr1,
                neighbor_list2,
                neighbor_ptr2,
            )
        else:
            return (
                neighbor_matrix1,
                num_neighbors1,
                neighbor_matrix2,
                num_neighbors2,
            )
    else:
        _naive_neighbor_matrix_pbc_dual_cutoff(
            positions=positions,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            cell=cell,
            neighbor_matrix1=neighbor_matrix1,
            neighbor_matrix2=neighbor_matrix2,
            neighbor_matrix_shifts1=neighbor_matrix_shifts1,
            neighbor_matrix_shifts2=neighbor_matrix_shifts2,
            num_neighbors1=num_neighbors1,
            num_neighbors2=num_neighbors2,
            shift_range_per_dimension=shift_range_per_dimension,
            shift_offset=shift_offset,
            total_shifts=total_shifts,
            half_fill=half_fill,
        )
        if return_neighbor_list:
            neighbor_list1, neighbor_ptr1, unit_shifts1 = (
                get_neighbor_list_from_neighbor_matrix(
                    neighbor_matrix1,
                    num_neighbors=num_neighbors1,
                    neighbor_shift_matrix=neighbor_matrix_shifts1,
                    fill_value=fill_value,
                )
            )
            neighbor_list2, neighbor_ptr2, unit_shifts2 = (
                get_neighbor_list_from_neighbor_matrix(
                    neighbor_matrix2,
                    num_neighbors=num_neighbors2,
                    neighbor_shift_matrix=neighbor_matrix_shifts2,
                    fill_value=fill_value,
                )
            )
            return (
                neighbor_list1,
                neighbor_ptr1,
                unit_shifts1,
                neighbor_list2,
                neighbor_ptr2,
                unit_shifts2,
            )
        else:
            return (
                neighbor_matrix1,
                num_neighbors1,
                neighbor_matrix_shifts1,
                neighbor_matrix2,
                num_neighbors2,
                neighbor_matrix_shifts2,
            )


###########################################################################################
########################### Cell List PyTorch Bindings ####################################
###########################################################################################


def estimate_cell_list_sizes(
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    max_nbins: int = 1000,
) -> tuple[int, torch.Tensor]:
    """Estimate allocation sizes for torch.compile-friendly cell list construction.

    Provides conservative estimates for maximum memory allocations needed when
    building cell lists with fixed-size tensors to avoid dynamic allocation
    and graph breaks in torch.compile.

    This function is not torch.compile compatible because it returns an integer
    received from using torch.Tensor.item()

    Parameters
    ----------
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix defining the simulation box.
    pbc : torch.Tensor, shape (1, 3), dtype=bool
        Flags indicating periodic boundary conditions in x, y, z directions.
    cutoff : float
        Maximum distance for neighbor search, determines minimum cell size.
    max_nbins : int, default=1000
        Maximum number of cells to allocate.

    Returns
    -------
    max_total_cells : int
        Estimated maximum number of cells needed for spatial decomposition.
        For degenerate cells, returns the total number of atoms.
    neighbor_search_radius : torch.Tensor, shape (3,), dtype=int32
        Radius of neighboring cells to search in each dimension.

    Notes
    -----
    - Cell size is determined by the cutoff distance to ensure neighboring
      cells contain all potential neighbors. The estimation assumes roughly
      cubic cells and uniform atomic distribution.
    - Currently, only unit cells with a positive determinant (i.e. with
      positive volume) are supported. For non-periodic systems, pass an identity
      cell.

    See Also
    --------
    nvalchemiops.neighbors.cell_list.build_cell_list : Core warp launcher
    allocate_cell_list : Allocates tensors based on these estimates
    build_cell_list : High-level wrapper that uses these estimates
    """
    if cell.numel() > 0 and cell.det() <= 0.0:
        raise RuntimeError(
            "Cell with volume <= 0.0 detected and is not supported."
            " Please pass unit cells with `det(cell) > 0.0`."
        )
    dtype = cell.dtype
    device = cell.device
    wp_device = str(device)
    wp_dtype = get_wp_dtype(dtype)
    wp_mat_dtype = get_wp_mat_dtype(dtype)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)

    if (cell.ndim == 3 and cell.shape[0] == 0) or cutoff <= 0:
        return 1, torch.zeros((3,), dtype=torch.int32, device=device)

    if cell.ndim == 2:
        cell = cell.unsqueeze(0)

    max_total_cells = torch.zeros(1, device=device, dtype=torch.int32)
    wp_max_total_cells = wp.from_torch(
        max_total_cells, dtype=wp.int32, return_ctype=True
    )

    neighbor_search_radius = torch.zeros((3,), dtype=torch.int32, device=device)
    wp_neighbor_search_radius = wp.from_torch(
        neighbor_search_radius, dtype=wp.int32, return_ctype=True
    )

    # Note: Using the _estimate_cell_list_sizes kernel from cell_list module
    from nvalchemiops.neighbors.cell_list import _estimate_cell_list_sizes_overload

    wp.launch(
        _estimate_cell_list_sizes_overload[wp_dtype],
        dim=1,
        inputs=[
            wp_cell,
            wp_pbc,
            wp_dtype(cutoff),
            max_nbins,
            wp_max_total_cells,
            wp_neighbor_search_radius,
        ],
        device=wp_device,
    )

    return (
        max_total_cells.item(),
        neighbor_search_radius,
    )


@torch.library.custom_op(
    "nvalchemiops::build_cell_list",
    mutates_args=(
        "cells_per_dimension",
        "atom_periodic_shifts",
        "atom_to_cell_mapping",
        "atoms_per_cell_count",
        "cell_atom_start_indices",
        "cell_atom_list",
    ),
)
def _build_cell_list_op(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
) -> None:
    """Internal custom op for building spatial cell list.

    This function is torch compilable.

    Notes
    -----
    The neighbor_search_radius is not an input parameter because it's computed
    internally by the warp launcher and doesn't need to be passed in.

    See Also
    --------
    nvalchemiops.neighbors.cell_list.build_cell_list : Core warp launcher
    build_cell_list : High-level wrapper function
    """
    total_atoms = positions.shape[0]
    device = positions.device

    # Handle empty case
    if total_atoms == 0:
        return

    cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
    pbc = pbc.squeeze(0)

    # Get warp dtypes and arrays
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    wp_device = str(device)

    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)

    wp_cells_per_dimension = wp.from_torch(
        cells_per_dimension, dtype=wp.int32, return_ctype=True
    )
    wp_atom_periodic_shifts = wp.from_torch(
        atom_periodic_shifts, dtype=wp.vec3i, return_ctype=True
    )
    wp_atom_to_cell_mapping = wp.from_torch(
        atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
    )
    # underlying warp launcher relies on Python API for array_scan
    # so `return_ctype` is omitted
    wp_atoms_per_cell_count = wp.from_torch(atoms_per_cell_count, dtype=wp.int32)
    wp_cell_atom_start_indices = wp.from_torch(cell_atom_start_indices, dtype=wp.int32)
    wp_cell_atom_list = wp.from_torch(cell_atom_list, dtype=wp.int32, return_ctype=True)

    # Zero atoms_per_cell_count before building
    atoms_per_cell_count.zero_()

    # Call core warp launcher
    wp_build_cell_list(
        positions=wp_positions,
        cell=wp_cell,
        pbc=wp_pbc,
        cutoff=cutoff,
        cells_per_dimension=wp_cells_per_dimension,
        atom_periodic_shifts=wp_atom_periodic_shifts,
        atom_to_cell_mapping=wp_atom_to_cell_mapping,
        atoms_per_cell_count=wp_atoms_per_cell_count,
        cell_atom_start_indices=wp_cell_atom_start_indices,
        cell_atom_list=wp_cell_atom_list,
        wp_dtype=wp_dtype,
        device=wp_device,
    )

    # Compute cell atom start indices using cumsum
    max_total_cells = atoms_per_cell_count.shape[0]
    cell_atom_start_indices[0] = 0
    if max_total_cells > 1:
        torch.cumsum(atoms_per_cell_count[:-1], dim=0, out=cell_atom_start_indices[1:])


def build_cell_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
) -> None:
    """Build spatial cell list with fixed allocation sizes for torch.compile compatibility.

    Constructs a spatial decomposition data structure for efficient neighbor searching.
    Uses fixed-size memory allocations to prevent dynamic tensor creation that would
    cause graph breaks in torch.compile.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Atomic coordinates in Cartesian space where total_atoms is the number of atoms.
        Must be float32, float64, or float16 dtype.
    cutoff : float
        Maximum distance for neighbor search. Determines minimum cell size.
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix defining the simulation box. Each row represents a
        lattice vector in Cartesian coordinates. Must match positions dtype.
    pbc : torch.Tensor, shape (3,), dtype=bool
        Flags indicating periodic boundary conditions in x, y, z directions.
        True enables PBC, False disables it for that dimension.
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32
        OUTPUT: Number of cells created in x, y, z directions.
    neighbor_search_radius : torch.Tensor, shape (3,), dtype=int32
        Radius of neighboring cells to search in each dimension. Passed through
        from allocate_cell_list for API continuity but not used in this function.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32
        OUTPUT: Periodic boundary crossings for each atom.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        OUTPUT: 3D cell coordinates assigned to each atom.
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32
        OUTPUT: Number of atoms in each cell. Only first 'total_cells' entries are valid.
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32
        OUTPUT: Starting index in cell_atom_list for each cell's atoms.
    cell_atom_list : torch.Tensor, shape (total_atoms,), dtype=int32
        OUTPUT: Flattened list of atom indices organized by cell. Use with start_indices
        to extract atoms for each cell.

    Notes
    -----
    - This function is torch.compile compatible and uses only static tensor shapes
    - Memory usage is determined by max_total_cells
    - For optimal performance, use estimates from estimate_cell_list_sizes()
    - Cell list must be rebuilt when atoms move between cells or PBC/cell changes

    See Also
    --------
    nvalchemiops.neighbors.cell_list.build_cell_list : Core warp launcher
    estimate_cell_list_sizes : Estimate memory requirements
    query_cell_list : Query the built cell list for neighbors
    cell_list : High-level function that builds and queries in one call
    """
    return _build_cell_list_op(
        positions,
        cutoff,
        cell,
        pbc,
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
    )


@torch.library.custom_op(
    "nvalchemiops::query_cell_list",
    mutates_args=("neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"),
)
def _query_cell_list_op(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    half_fill: bool = False,
) -> None:
    """Internal custom op for querying spatial cell list to build neighbor matrix.

    This function is torch compilable.

    See Also
    --------
    nvalchemiops.neighbors.cell_list.query_cell_list : Core warp launcher
    query_cell_list : High-level wrapper function
    """
    total_atoms = positions.shape[0]
    device = positions.device

    # Handle empty case
    if total_atoms == 0:
        return

    cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
    pbc = pbc.squeeze(0)

    # Get warp dtypes and arrays
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    wp_device = str(device)

    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)

    wp_cells_per_dimension = wp.from_torch(
        cells_per_dimension, dtype=wp.int32, return_ctype=True
    )
    wp_neighbor_search_radius = wp.from_torch(
        neighbor_search_radius, dtype=wp.int32, return_ctype=True
    )
    wp_atom_periodic_shifts = wp.from_torch(
        atom_periodic_shifts, dtype=wp.vec3i, return_ctype=True
    )
    wp_atom_to_cell_mapping = wp.from_torch(
        atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
    )
    wp_atoms_per_cell_count = wp.from_torch(
        atoms_per_cell_count, dtype=wp.int32, return_ctype=True
    )
    wp_cell_atom_start_indices = wp.from_torch(
        cell_atom_start_indices, dtype=wp.int32, return_ctype=True
    )
    wp_cell_atom_list = wp.from_torch(cell_atom_list, dtype=wp.int32, return_ctype=True)

    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, return_ctype=True
    )
    wp_neighbor_matrix_shifts = wp.from_torch(
        neighbor_matrix_shifts, dtype=wp.vec3i, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True)

    # Call core warp launcher
    wp_query_cell_list(
        positions=wp_positions,
        cell=wp_cell,
        pbc=wp_pbc,
        cutoff=cutoff,
        cells_per_dimension=wp_cells_per_dimension,
        neighbor_search_radius=wp_neighbor_search_radius,
        atom_periodic_shifts=wp_atom_periodic_shifts,
        atom_to_cell_mapping=wp_atom_to_cell_mapping,
        atoms_per_cell_count=wp_atoms_per_cell_count,
        cell_atom_start_indices=wp_cell_atom_start_indices,
        cell_atom_list=wp_cell_atom_list,
        neighbor_matrix=wp_neighbor_matrix,
        neighbor_matrix_shifts=wp_neighbor_matrix_shifts,
        num_neighbors=wp_num_neighbors,
        wp_dtype=wp_dtype,
        device=wp_device,
        half_fill=half_fill,
    )


def query_cell_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    half_fill: bool = False,
) -> None:
    """Query spatial cell list to build neighbor matrix with distance constraints.

    Uses pre-built cell list data structures to efficiently find all atom pairs
    within the specified cutoff distance. Handles periodic boundary conditions
    and returns neighbor matrix format.

    This function is torch compilable.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Atomic coordinates in Cartesian space.
    cutoff : float
        Maximum distance for considering atoms as neighbors.
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix for periodic boundary coordinate shifts.
    pbc : torch.Tensor, shape (3,), dtype=bool
        Periodic boundary condition flags.
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32
        Number of cells in x, y, z directions from build_cell_list.
    neighbor_search_radius : torch.Tensor, shape (3,), dtype=int32
        Shifts to search from build_cell_list.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32
        Periodic boundary crossings for each atom from build_cell_list.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32
        3D cell coordinates for each atom from build_cell_list.
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32
        Number of atoms in each cell from build_cell_list.
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32
        Starting index in cell_atom_list for each cell from build_cell_list.
    cell_atom_list : torch.Tensor, shape (total_atoms,), dtype=int32
        Flattened list of atom indices organized by cell from build_cell_list.
    neighbor_matrix : torch.Tensor, shape (total_atoms, max_neighbors), dtype=int32
        OUTPUT: Neighbor matrix to be filled with neighbor atom indices.
        Must be pre-allocated.
    neighbor_matrix_shifts : torch.Tensor, shape (total_atoms, max_neighbors, 3), dtype=int32
        OUTPUT: Matrix storing shift vectors for each neighbor relationship.
        Must be pre-allocated.
    num_neighbors : torch.Tensor, shape (total_atoms,), dtype=int32
        OUTPUT: Number of neighbors found for each atom.
        Must be pre-allocated.
    half_fill : bool, default=False
        If True, only store half of the neighbor relationships.

    See Also
    --------
    nvalchemiops.neighbors.cell_list.query_cell_list : Core warp launcher
    build_cell_list : Builds the cell list data structures
    cell_list : High-level function that builds and queries in one call
    """
    return _query_cell_list_op(
        positions,
        cutoff,
        cell,
        pbc,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        half_fill,
    )


def cell_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    max_neighbors: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    num_neighbors: torch.Tensor | None = None,
    cells_per_dimension: torch.Tensor | None = None,
    neighbor_search_radius: torch.Tensor | None = None,
    atom_periodic_shifts: torch.Tensor | None = None,
    atom_to_cell_mapping: torch.Tensor | None = None,
    atoms_per_cell_count: torch.Tensor | None = None,
    cell_atom_start_indices: torch.Tensor | None = None,
    cell_atom_list: torch.Tensor | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor]
):
    """Build complete neighbor matrix using spatial cell list acceleration.

    High-level convenience function that automatically estimates memory requirements,
    builds spatial cell list data structures, and queries them to produce a complete
    neighbor matrix. Combines build_cell_list and query_cell_list operations.

    Parameters
    ----------
    positions : torch.Tensor, shape (total_atoms, 3)
        Atomic coordinates in Cartesian space where total_atoms is the number of atoms.
    cutoff : float
        Maximum distance for neighbor search.
    cell : torch.Tensor, shape (1, 3, 3)
        Unit cell matrix defining the simulation box. Each row represents a
        lattice vector in Cartesian coordinates.
    pbc : torch.Tensor, shape (1, 3), dtype=bool
        Flags indicating periodic boundary conditions in x, y, z directions.
    max_neighbors : int, optional
        Maximum number of neighbors per atom. If not provided, will be estimated automatically.
    half_fill : bool, optional
        If True, only fill half of the neighbor matrix. Default is False.
    fill_value : int | None, optional
        Value to fill the neighbor matrix with. Default is total_atoms.
    return_neighbor_list : bool, optional - default = False
        If True, convert the neighbor matrix to a neighbor list (idx_i, idx_j) format by
        creating a mask over the fill_value, which can incur a performance penalty.
        We recommend using the neighbor matrix format,
        and only convert to a neighbor list format if absolutely necessary.
    neighbor_matrix : torch.Tensor, optional
        Pre-allocated tensor of shape (total_atoms, max_neighbors) for neighbor indices.
        If None, allocated internally.
    neighbor_matrix_shifts : torch.Tensor, optional
        Pre-allocated tensor of shape (total_atoms, max_neighbors, 3) for shift vectors.
        If None, allocated internally.
    num_neighbors : torch.Tensor, optional
        Pre-allocated tensor of shape (total_atoms,) for neighbor counts.
        If None, allocated internally.
    cells_per_dimension : torch.Tensor, shape (3,), dtype=int32, optional
        Number of cells in x, y, z directions.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    neighbor_search_radius : torch.Tensor, shape (3,), dtype=int32, optional
        Radius of neighboring cells to search in each dimension.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    atom_periodic_shifts : torch.Tensor, shape (total_atoms, 3), dtype=int32, optional
        Periodic boundary crossings for each atom.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    atom_to_cell_mapping : torch.Tensor, shape (total_atoms, 3), dtype=int32, optional
        Cell coordinates for each atom.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    atoms_per_cell_count : torch.Tensor, shape (max_total_cells,), dtype=int32, optional
        Number of atoms in each cell.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    cell_atom_start_indices : torch.Tensor, shape (max_total_cells,), dtype=int32, optional
        Starting index in cell_atom_list for each cell.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.
    cell_atom_list : torch.Tensor, shape (total_atoms,), dtype=int32, optional
        Flattened list of atom indices organized by cell.
        Pass a pre-allocated tensor to avoid reallocation for cell list construction.
        If None, allocated internally to build the cell list.

    Returns
    -------
    results : tuple of torch.Tensor
        Variable-length tuple depending on input parameters. The return pattern follows:

        - Matrix format (default): ``(neighbor_matrix, num_neighbors, neighbor_matrix_shifts)``
        - List format (return_neighbor_list=True): ``(neighbor_list, neighbor_ptr, neighbor_list_shifts)``

    Notes
    -----
    - This is the main user-facing API for cell list neighbor construction
    - Uses automatic memory allocation estimation for torch.compile compatibility
    - For advanced users who want to cache cell lists, use build_cell_list and query_cell_list separately
    - Returns appropriate empty tensors for systems with <= 1 atom or cutoff <= 0

    See Also
    --------
    nvalchemiops.neighbors.cell_list.build_cell_list : Core warp launcher for building
    nvalchemiops.neighbors.cell_list.query_cell_list : Core warp launcher for querying
    naive_neighbor_list : O(N²) method for small systems
    """
    total_atoms = positions.shape[0]
    device = positions.device
    cell = cell if cell.ndim == 3 else cell.unsqueeze(0)
    pbc = pbc.squeeze(0)

    if fill_value is None:
        fill_value = total_atoms

    # Handle empty case
    if total_atoms <= 0 or cutoff <= 0:
        if return_neighbor_list:
            return (
                torch.zeros((2, 0), dtype=torch.int32, device=device),
                torch.zeros((total_atoms + 1,), dtype=torch.int32, device=device),
                torch.zeros((0, 3), dtype=torch.int32, device=device),
            )
        else:
            return (
                torch.full(
                    (total_atoms, 0), fill_value, dtype=torch.int32, device=device
                ),
                torch.zeros((total_atoms,), dtype=torch.int32, device=device),
                torch.zeros((total_atoms, 0, 3), dtype=torch.int32, device=device),
            )

    if max_neighbors is None and (
        neighbor_matrix is None
        or neighbor_matrix_shifts is None
        or num_neighbors is None
    ):
        max_neighbors = estimate_max_neighbors(cutoff)

    if neighbor_matrix is None:
        neighbor_matrix = torch.full(
            (total_atoms, max_neighbors), fill_value, dtype=torch.int32, device=device
        )
    else:
        neighbor_matrix.fill_(fill_value)
    if neighbor_matrix_shifts is None:
        neighbor_matrix_shifts = torch.zeros(
            (total_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )
    else:
        neighbor_matrix_shifts.zero_()
    if num_neighbors is None:
        num_neighbors = torch.zeros((total_atoms,), dtype=torch.int32, device=device)
    else:
        num_neighbors.zero_()

    # Allocate cell list if needed
    if (
        cells_per_dimension is None
        or neighbor_search_radius is None
        or atom_periodic_shifts is None
        or atom_to_cell_mapping is None
        or atoms_per_cell_count is None
        or cell_atom_start_indices is None
        or cell_atom_list is None
    ):
        max_total_cells, neighbor_search_radius = estimate_cell_list_sizes(
            cell, pbc, cutoff
        )
        cell_list_cache = allocate_cell_list(
            total_atoms,
            max_total_cells,
            neighbor_search_radius,
            device,
        )
    else:
        cells_per_dimension.zero_()
        atom_periodic_shifts.zero_()
        atom_to_cell_mapping.zero_()
        atoms_per_cell_count.zero_()
        cell_atom_start_indices.zero_()
        cell_atom_list.zero_()
        cell_list_cache = (
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        )

    build_cell_list(
        positions,
        cutoff,
        cell,
        pbc,
        *cell_list_cache,
    )

    query_cell_list(
        positions,
        cutoff,
        cell,
        pbc,
        *cell_list_cache,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        half_fill,
    )

    if return_neighbor_list:
        neighbor_list, neighbor_ptr, neighbor_list_shifts = (
            get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix,
                num_neighbors=num_neighbors,
                neighbor_shift_matrix=neighbor_matrix_shifts,
                fill_value=fill_value,
            )
        )
        return neighbor_list, neighbor_ptr, neighbor_list_shifts
    else:
        return neighbor_matrix, num_neighbors, neighbor_matrix_shifts
