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

"""Tests for PyTorch bindings of naive dual cutoff neighbor list methods."""

import pytest
import torch

from nvalchemiops.torch.neighbors.naive_dual_cutoff import (
    naive_neighbor_list_dual_cutoff,
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    compute_naive_num_shifts,
)

from ...test_utils import (
    assert_neighbor_lists_equal,
    brute_force_neighbors,
    create_simple_cubic_system,
)
from .conftest import requires_vesin


class TestNaiveDualCutoffMainAPI:
    """Test the main naive dual cutoff neighbor list API function."""

    @requires_vesin
    @pytest.mark.parametrize("half_fill", [False, True])
    @pytest.mark.parametrize("pbc_flag", [True, False])
    @pytest.mark.parametrize("preallocate", [False, True])
    @pytest.mark.parametrize("return_neighbor_list", [False, True])
    @pytest.mark.parametrize("fill_value", [-1, 8])
    def test_naive_neighbor_list_dual_cutoff_no_pbc(
        self,
        device,
        dtype,
        half_fill,
        pbc_flag,
        preallocate,
        return_neighbor_list,
        fill_value,
    ):
        """Test naive_neighbor_list_dual_cutoff without periodic boundary conditions."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 15
        max_neighbors2 = 25

        if not pbc_flag:
            cell = None
            pbc = None
        if preallocate:
            neighbor_matrix1 = torch.full(
                (positions.shape[0], max_neighbors1),
                fill_value,
                dtype=torch.int32,
                device=device,
            )
            num_neighbors1 = torch.zeros(
                positions.shape[0], dtype=torch.int32, device=device
            )
            neighbor_matrix2 = torch.full(
                (positions.shape[0], max_neighbors2),
                fill_value,
                dtype=torch.int32,
                device=device,
            )
            num_neighbors2 = torch.zeros(
                positions.shape[0], dtype=torch.int32, device=device
            )
            args = (positions, cutoff1, cutoff2)
            kwargs = {
                "fill_value": fill_value,
                "half_fill": half_fill,
                "neighbor_matrix1": neighbor_matrix1,
                "num_neighbors1": num_neighbors1,
                "neighbor_matrix2": neighbor_matrix2,
                "num_neighbors2": num_neighbors2,
                "return_neighbor_list": return_neighbor_list,
            }
            if pbc_flag:
                shift_range_per_dimension, shift_offset, total_shifts = (
                    compute_naive_num_shifts(cell, cutoff2, pbc)
                )
                kwargs["cell"] = cell
                kwargs["pbc"] = pbc
                kwargs["shift_range_per_dimension"] = shift_range_per_dimension
                kwargs["shift_offset"] = shift_offset
                kwargs["total_shifts"] = total_shifts
                neighbor_matrix_shifts1 = torch.zeros(
                    (positions.shape[0], max_neighbors1, 3),
                    dtype=torch.int32,
                    device=device,
                )
                kwargs["neighbor_matrix_shifts1"] = neighbor_matrix_shifts1
                neighbor_matrix_shifts2 = torch.zeros(
                    (positions.shape[0], max_neighbors2, 3),
                    dtype=torch.int32,
                    device=device,
                )
                kwargs["neighbor_matrix_shifts2"] = neighbor_matrix_shifts2
            results = naive_neighbor_list_dual_cutoff(*args, **kwargs)
            if return_neighbor_list:
                if pbc_flag:
                    (
                        neighbor_list1,
                        neighbor_ptr1,
                        neighbor_shifts1,
                        neighbor_list2,
                        neighbor_ptr2,
                        neighbor_shifts2,
                    ) = results
                    num_neighbors1 = neighbor_ptr1[1:] - neighbor_ptr1[:-1]
                    num_neighbors2 = neighbor_ptr2[1:] - neighbor_ptr2[:-1]
                    idx_i1 = neighbor_list1[0]
                    idx_j1 = neighbor_list1[1]
                    idx_i2 = neighbor_list2[0]
                    idx_j2 = neighbor_list2[1]
                    u1 = neighbor_shifts1
                    u2 = neighbor_shifts2
                else:
                    (
                        neighbor_list1,
                        neighbor_ptr1,
                        neighbor_list2,
                        neighbor_ptr2,
                    ) = results
                    num_neighbors1 = neighbor_ptr1[1:] - neighbor_ptr1[:-1]
                    num_neighbors2 = neighbor_ptr2[1:] - neighbor_ptr2[:-1]
                    idx_i1 = neighbor_list1[0]
                    idx_j1 = neighbor_list1[1]
                    idx_i2 = neighbor_list2[0]
                    idx_j2 = neighbor_list2[1]
                    u1 = torch.zeros(
                        (idx_i1.shape[0], 3), dtype=torch.int32, device=device
                    )
                    u2 = torch.zeros(
                        (idx_j2.shape[0], 3), dtype=torch.int32, device=device
                    )
        else:
            args = (positions, cutoff1, cutoff2)
            kwargs = {
                "max_neighbors1": max_neighbors1,
                "max_neighbors2": max_neighbors2,
                "fill_value": fill_value,
                "half_fill": half_fill,
                "return_neighbor_list": return_neighbor_list,
            }
            if pbc_flag:
                kwargs["cell"] = cell
                kwargs["pbc"] = pbc
            results = naive_neighbor_list_dual_cutoff(*args, **kwargs)
            if pbc_flag:
                if return_neighbor_list:
                    (
                        neighbor_list1,
                        neighbor_ptr1,
                        neighbor_shifts1,
                        neighbor_list2,
                        neighbor_ptr2,
                        neighbor_shifts2,
                    ) = results
                    num_neighbors1 = neighbor_ptr1[1:] - neighbor_ptr1[:-1]
                    num_neighbors2 = neighbor_ptr2[1:] - neighbor_ptr2[:-1]
                    idx_i1 = neighbor_list1[0]
                    idx_j1 = neighbor_list1[1]
                    idx_i2 = neighbor_list2[0]
                    idx_j2 = neighbor_list2[1]
                    u1 = neighbor_shifts1
                    u2 = neighbor_shifts2
                else:
                    (
                        neighbor_matrix1,
                        num_neighbors1,
                        neighbor_matrix_shifts1,
                        neighbor_matrix2,
                        num_neighbors2,
                        neighbor_matrix_shifts2,
                    ) = results
            else:
                if return_neighbor_list:
                    (
                        neighbor_list1,
                        neighbor_ptr1,
                        neighbor_list2,
                        neighbor_ptr2,
                    ) = results
                    num_neighbors1 = neighbor_ptr1[1:] - neighbor_ptr1[:-1]
                    num_neighbors2 = neighbor_ptr2[1:] - neighbor_ptr2[:-1]
                    idx_i1 = neighbor_list1[0]
                    idx_j1 = neighbor_list1[1]
                    idx_i2 = neighbor_list2[0]
                    idx_j2 = neighbor_list2[1]
                    u1 = torch.zeros(
                        (idx_i1.shape[0], 3), dtype=torch.int32, device=device
                    )
                    u2 = torch.zeros(
                        (idx_j2.shape[0], 3), dtype=torch.int32, device=device
                    )
                else:
                    (
                        neighbor_matrix1,
                        num_neighbors1,
                        neighbor_matrix2,
                        num_neighbors2,
                    ) = results

        # Get reference result
        i_ref1, j_ref1, u_ref1, _ = brute_force_neighbors(positions, cell, pbc, cutoff1)
        i_ref2, j_ref2, u_ref2, _ = brute_force_neighbors(positions, cell, pbc, cutoff2)

        if return_neighbor_list and not half_fill:
            assert_neighbor_lists_equal(
                (idx_i1, idx_j1, u1),
                (i_ref1, j_ref1, u_ref1),
            )
            assert_neighbor_lists_equal(
                (idx_i2, idx_j2, u2),
                (i_ref2, j_ref2, u_ref2),
            )

    def test_naive_neighbor_list_dual_cutoff_edge_cases(
        self,
        device,
        dtype,
    ):
        """Test edge cases for naive_neighbor_list_dual_cutoff."""
        # Empty system
        positions_empty = torch.empty(0, 3, dtype=dtype, device=device)
        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            naive_neighbor_list_dual_cutoff(
                positions=positions_empty,
                cutoff1=1.0,
                cutoff2=1.5,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=None,
                cell=None,
            )
        )
        assert neighbor_matrix1.shape == (0, 10)
        assert neighbor_matrix2.shape == (0, 15)
        assert num_neighbors1.shape == (0,)
        assert num_neighbors2.shape == (0,)

        # Single atom
        positions_single = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            naive_neighbor_list_dual_cutoff(
                positions=positions_single,
                cutoff1=1.0,
                cutoff2=1.5,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=None,
                cell=None,
            )
        )
        assert num_neighbors1[0].item() == 0, "Single atom should have no neighbors"
        assert num_neighbors2[0].item() == 0, "Single atom should have no neighbors"

        # Zero cutoffs
        positions, _, _ = create_simple_cubic_system(
            num_atoms=4, dtype=dtype, device=device
        )
        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            naive_neighbor_list_dual_cutoff(
                positions=positions,
                cutoff1=0.0,
                cutoff2=0.0,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=None,
                cell=None,
            )
        )
        assert torch.all(num_neighbors1 == 0), "Zero cutoffs should find no neighbors"
        assert torch.all(num_neighbors2 == 0), "Zero cutoffs should find no neighbors"

    @pytest.mark.parametrize("half_fill", [True, False])
    def test_naive_neighbor_list_dual_cutoff_error_conditions(
        self, device, dtype, half_fill
    ):
        """Test error conditions for naive_neighbor_list_dual_cutoff."""
        positions, cell, pbc = create_simple_cubic_system(dtype=dtype, device=device)

        # Test mismatched cell and pbc arguments
        with pytest.raises(
            ValueError, match="If cell is provided, pbc must also be provided"
        ):
            naive_neighbor_list_dual_cutoff(
                positions, 1.0, 1.5, pbc=None, cell=cell, max_neighbors1=10
            )

        with pytest.raises(
            ValueError, match="If pbc is provided, cell must also be provided"
        ):
            naive_neighbor_list_dual_cutoff(
                positions, 1.0, 1.5, pbc=pbc, cell=None, max_neighbors1=10
            )

    def test_max_neighbors2_default(self):
        """Test that max_neighbors2 defaults to max_neighbors1 when not provided."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
        )

        # Call without max_neighbors2
        result = naive_neighbor_list_dual_cutoff(
            positions=positions,
            cutoff1=0.5,
            cutoff2=1.5,
            max_neighbors1=10,
            # max_neighbors2 not provided
        )

        # Should not raise an error and should work correctly
        assert (
            len(result) == 4
        )  # neighbor_matrix1, neighbor_matrix2, num_neighbors1, num_neighbors2
