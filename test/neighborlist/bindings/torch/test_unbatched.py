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

"""Tests for PyTorch bindings of unbatched neighbor list methods."""

from importlib import import_module

import pytest
import torch

from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.torch.neighbors.neighbor_utils import (
    allocate_cell_list,
    compute_naive_num_shifts,
)
from nvalchemiops.torch.neighbors.unbatched import (
    build_cell_list,
    cell_list,
    estimate_cell_list_sizes,
    naive_neighbor_list,
    naive_neighbor_list_dual_cutoff,
    query_cell_list,
)

from ...test_utils import (
    assert_neighbor_lists_equal,
    brute_force_neighbors,
    create_nonorthorhombic_system,
    create_random_system,
    create_simple_cubic_system,
)

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda:0")
dtypes = [torch.float32, torch.float64]

try:
    _ = import_module("vesin")
    run_vesin_checks = True
except ModuleNotFoundError:
    run_vesin_checks = False


class TestCellListAPI:
    """Test the main cell list API functions."""

    @pytest.mark.skipif(
        not run_vesin_checks, reason="`vesin` required for consistency checks."
    )
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_single_atom_system(self, device, dtype):
        """Test with single atom (should have no neighbors)."""
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        cell = (torch.eye(3, dtype=dtype, device=device) * 2.0).reshape(1, 3, 3)
        pbc = torch.tensor([True, True, True], device=device)
        cutoff = 3.0

        neighbor_list, _, u = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )
        i, j = neighbor_list
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)

        # Results should be identical
        assert_neighbor_lists_equal((i, j, u), (i_ref, j_ref, u_ref))

    @pytest.mark.skipif(
        not run_vesin_checks, reason="`vesin` required for consistency checks."
    )
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_two_atom_system(self, device, dtype):
        """Test simple two-atom system."""
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype, device=device
        )
        cell = (torch.eye(3, dtype=dtype, device=device) * 2.0).reshape(1, 3, 3)
        pbc = torch.tensor([True, True, True], device=device)
        cutoff = 1.0

        neighbor_list, _, u = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )
        i, j = neighbor_list
        assert len(i) == 2, f"Expected 2 neighbors, got {len(i)}"

        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        assert_neighbor_lists_equal((i, j, u), (i_ref, j_ref, u_ref))

    @pytest.mark.skipif(
        not run_vesin_checks, reason="`vesin` required for consistency checks."
    )
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_cubic_system(self, device, dtype):
        """Test with simple cubic lattice."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        cutoff = 1.1  # Should capture nearest neighbors

        neighbor_list, _, u = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )
        i, j = neighbor_list

        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        assert_neighbor_lists_equal((i, j, u), (i_ref, j_ref, u_ref))

    @pytest.mark.skipif(
        not run_vesin_checks, reason="`vesin` required for consistency checks."
    )
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("pbc_flag", [True, False])
    def test_random_system(self, device, dtype, pbc_flag):
        """Test with random atomic positions."""
        positions, cell, pbc = create_random_system(
            num_atoms=20,
            cell_size=10.0,
            dtype=dtype,
            device=device,
            seed=42,
            pbc_flag=pbc_flag,
        )
        cutoff = 5.0

        neighbor_list, _, u = cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            max_neighbors=1500,
            return_neighbor_list=True,
        )
        i, j = neighbor_list
        ref_i, ref_j, ref_u, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        assert_neighbor_lists_equal((i, j, u), (ref_i, ref_j, ref_u))

        # Check consistency: if (i,j) is a pair, j should be within cutoff of i
        if len(i) > 0:
            for idx in range(min(10, len(i))):  # Check first 10 pairs
                atom_i, atom_j = i[idx].item(), j[idx].item()
                shift = cell.squeeze(0) @ u[idx].to(dtype)
                rij = positions[atom_j] - positions[atom_i] + shift
                dist = torch.norm(rij, dim=0).item()
                assert dist < cutoff + 1e-5, f"Distance {dist} exceeds cutoff {cutoff}"

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_no_pbc(self, device, dtype, return_neighbor_list):
        """Test with no periodic boundary conditions."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=3.0, dtype=dtype, device=device
        )
        pbc = torch.tensor([False, False, False], device=device)
        cutoff = 3.0

        results = cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            return_neighbor_list=return_neighbor_list,
        )
        u = results[-1]

        # With no PBC, all shifts should be zero
        if len(u) > 0:
            assert torch.all(u == 0), "All shifts should be zero with no PBC"

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    @pytest.mark.parametrize("preallocate", [True, False])
    @pytest.mark.parametrize("fill_value", [None, -1])
    @pytest.mark.parametrize("cell_pbc_shape", [0, 1])
    def test_mixed_pbc(
        self,
        device,
        dtype,
        return_neighbor_list,
        preallocate,
        fill_value,
        cell_pbc_shape,
    ):
        """Test with mixed periodic boundary conditions."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        cutoff = 3.0

        if cell_pbc_shape == 0:
            cell = cell.reshape(3, 3)
            pbc = pbc.reshape(3)
        else:
            cell = cell.reshape(1, 3, 3)
            pbc = pbc.reshape(1, 3)

        if preallocate:
            max_neighbors = estimate_max_neighbors(cutoff)
            max_cells, neighbor_search_radius = estimate_cell_list_sizes(
                cell, pbc, cutoff
            )
            (
                cells_per_dimension,
                neighbor_search_radius,
                atom_periodic_shifts,
                atom_to_cell_mapping,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
            ) = allocate_cell_list(
                positions.shape[0], max_cells, neighbor_search_radius, device
            )
            fill_value = positions.shape[0] if fill_value is None else fill_value
            neighbor_matrix = torch.full(
                (positions.shape[0], max_neighbors),
                fill_value,
                dtype=torch.int32,
                device=device,
            )
            neighbor_matrix_shifts = torch.zeros(
                (positions.shape[0], max_neighbors, 3), dtype=torch.int32, device=device
            )
            num_neighbors = torch.zeros(
                (positions.shape[0],), dtype=torch.int32, device=device
            )

            results = cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                fill_value=fill_value,
                return_neighbor_list=return_neighbor_list,
                cells_per_dimension=cells_per_dimension,
                neighbor_search_radius=neighbor_search_radius,
                atom_periodic_shifts=atom_periodic_shifts,
                atom_to_cell_mapping=atom_to_cell_mapping,
                atoms_per_cell_count=atoms_per_cell_count,
                cell_atom_start_indices=cell_atom_start_indices,
                cell_atom_list=cell_atom_list,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                num_neighbors=num_neighbors,
            )
        else:
            results = cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                fill_value=fill_value,
                return_neighbor_list=return_neighbor_list,
            )

        if return_neighbor_list:
            neighbor_list, _, u = results
            assert len(neighbor_list) == 2
            assert u[:, 2].sum().item() == 0
            assert (u[:, 0] ** 2).sum().item() > 0
        else:
            _, _, u = results
            assert u[:, :, 2].sum().item() == 0
            assert (u[:, :, 0] ** 2).sum().item() > 0

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_large_cutoff(self, device, dtype, return_neighbor_list):
        """Test with large cutoff that includes many neighbors."""
        positions, cell, pbc = create_random_system(
            num_atoms=10, cell_size=2.0, dtype=dtype, device=device, seed=123
        )
        cutoff = 5.0  # Large cutoff

        results = cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            return_neighbor_list=return_neighbor_list,
        )
        if return_neighbor_list:
            num_pairs = results[0].shape[1]
        else:
            num_pairs = results[1].sum().item()
        assert num_pairs >= 0

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_zero_cutoff(self, device, dtype, return_neighbor_list):
        """Test with zero cutoff (should find no neighbors)."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 0.0

        results = cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            return_neighbor_list=return_neighbor_list,
        )
        if return_neighbor_list:
            assert len(results) == 3
            assert results[0].shape == (2, 0)  # neighbor_list
            assert results[1].shape == (9,)  # neighbor_ptr
            assert results[2].shape == (0, 3)  # shifts
        else:
            assert len(results) == 3
            assert results[0].shape[0] == 8
            assert results[1].sum().item() == 0

    @pytest.mark.skipif(
        not run_vesin_checks, reason="`vesin` required for consistency checks."
    )
    @pytest.mark.parametrize(
        "pbc_flag",
        [
            [True, True, True],
            [False, False, False],
            [True, False, True],
            [False, False, True],
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("num_atoms", [10, 20, 50, 100])
    @pytest.mark.parametrize("cutoff", [1.0, 3.0, 5.0])
    @pytest.mark.parametrize("system_type", ["random", "nonorthorhombic"])
    def test_scaling_correctness(
        self, pbc_flag, dtype, device, num_atoms, cutoff, system_type
    ):
        """Test with random atomic positions."""
        if system_type == "random":
            positions, cell, pbc = create_random_system(
                num_atoms=num_atoms,
                cell_size=3.0,
                dtype=dtype,
                device=device,
                seed=42,
                pbc_flag=pbc_flag,
            )

        else:
            positions, cell, pbc = create_nonorthorhombic_system(
                num_atoms=num_atoms,
                a=8.57,
                b=12.9645,
                c=7.2203,
                alpha=90.74,
                beta=115.944,
                gamma=87.663,
                dtype=dtype,
                device=device,
                seed=42,
                pbc_flag=pbc_flag,
            )
            scale_factor = (1.0 / 720.88) ** (1.0 / 3.0)
            cell = cell * scale_factor
            positions = positions * scale_factor

        estimated_density = num_atoms / cell.det().abs().item()
        max_neighbors = estimate_max_neighbors(
            cutoff, atomic_density=estimated_density, safety_factor=5.0
        )
        neighbor_list, _, u = cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            max_neighbors=max_neighbors,
            return_neighbor_list=True,
        )
        i, j = neighbor_list
        # Basic checks
        ref_i, ref_j, ref_u, _ = brute_force_neighbors(positions, cell, pbc, cutoff)
        assert_neighbor_lists_equal((i, j, u), (ref_i, ref_j, ref_u))

        # Check consistency: if (i,j) is a pair, j should be within cutoff of i
        if len(i) > 0:
            for idx in range(min(10, len(i))):  # Check first 10 pairs
                atom_i, atom_j = i[idx].item(), j[idx].item()
                shift = cell.squeeze(0) @ u[idx].to(dtype)
                rij = positions[atom_j] - positions[atom_i] + shift
                dist = torch.norm(rij, dim=0).item()
                assert dist < cutoff + 1e-5, f"Distance {dist} exceeds cutoff {cutoff}"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_empty_estimate_cell_list_sizes(self, device, dtype):
        """Test that estimate_cell_list_sizes returns the correct values for an empty batch."""
        cell = torch.zeros((0, 3, 3), dtype=dtype, device=device)
        pbc = torch.zeros((0, 3), dtype=torch.bool, device=device)
        cutoff = 1.0
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(cell, pbc, cutoff)
        assert max_cells == 1
        assert neighbor_search_radius.shape == (3,)
        assert neighbor_search_radius.dtype == torch.int32
        assert neighbor_search_radius.device == torch.device(device)

        # Now test with negative cutoff
        cell = torch.eye(3, dtype=dtype, device=device).reshape(1, 3, 3)
        pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
        cutoff = -1.0
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(cell, pbc, cutoff)
        assert max_cells == 1
        assert neighbor_search_radius.shape == (3,)
        assert neighbor_search_radius.dtype == torch.int32
        assert neighbor_search_radius.device == torch.device(device)

    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_empty_system(self, return_neighbor_list):
        """Test with empty coordinate array."""
        positions = torch.empty(0, 3, dtype=torch.float32)
        cell = torch.eye(3, dtype=torch.float32)
        pbc = torch.tensor([True, True, True])
        cutoff = 1.0

        results = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=return_neighbor_list
        )
        if return_neighbor_list:
            assert len(results) == 3
            assert results[0].shape == (2, 0)  # neighbor_list
            assert results[1].shape == (1,)  # neighbor_ptr
            assert results[2].shape == (0, 3)  # shifts
        else:
            assert len(results) == 3
            assert results[0].shape[0] == 0  # neighbor_matrix
            assert results[1].shape[0] == 0  # num_neighbors
            assert results[2].shape[0] == 0  # neighbor_matrix_shifts
            assert results[2].shape[2] == 3
            assert results[1].shape == (0,)

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_zero_volume(self, device, dtype):
        """Check to make sure bindings raises error for zero volume"""
        positions = torch.rand((4, 3), device=device, dtype=dtype)
        # cell has zero and negative volumes
        cells = torch.tensor(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            ],
            dtype=dtype,
            device=device,
        )
        pbc = torch.ones((1, 3), dtype=bool, device=device)
        with pytest.raises(RuntimeError, match="Cell with volume <= 0"):
            _ = cell_list(
                positions,
                3.0,
                cells,
                pbc,
            )

    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_dtype_consistency(self, dtype, return_neighbor_list):
        """Test that output dtypes are consistent with inputs."""
        positions = torch.randn(5, 3, dtype=dtype)
        cell = (torch.eye(3, dtype=dtype) * 2.0).reshape(1, 3, 3)
        pbc = torch.tensor([True, True, True], dtype=torch.bool)
        cutoff = 1.5

        results = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=return_neighbor_list
        )

        for result in results:
            assert result.dtype == torch.int32

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_device_consistency(self, device, return_neighbor_list):
        """Test that outputs are on the same device as inputs."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions = torch.randn(5, 3, device=device)
        cell = torch.eye(3, device=device).reshape(1, 3, 3) * 2.0
        pbc = torch.tensor([True, True, True], device=device)
        cutoff = 1.5

        results = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=return_neighbor_list
        )
        for result in results:
            assert result.device == torch.device(device)


class TestCellListComponentsAPI:
    """Test the new modular cell list API functions."""

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_build_and_query_cell_list(self, device, dtype):
        """Test building and querying cell list separately."""
        positions, cell, pbc = create_simple_cubic_system(dtype=dtype, device=device)
        cutoff = 1.1
        pbc = pbc.squeeze(0)
        # Get size estimates for build_cell_list
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(
            cell,
            pbc,
            cutoff,
        )
        max_neighbors = estimate_max_neighbors(cutoff)

        total_atoms = positions.shape[0]

        # Allocate memory for the cell list
        cell_list_cache = allocate_cell_list(
            total_atoms, max_cells, neighbor_search_radius, device
        )

        # Build cell list
        build_cell_list(positions, cutoff, cell, pbc, *cell_list_cache)

        assert cell_list_cache[0] is not None
        assert cell_list_cache[0].device == torch.device(device)
        assert cell_list_cache[0].dtype == torch.int32
        assert cell_list_cache[0].shape == (3,)

        # Query using the cell list
        assert max_neighbors > 0
        neighbor_matrix = torch.full(
            (total_atoms, max_neighbors),
            fill_value=-1,
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix_shifts = torch.zeros(
            (total_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros((total_atoms,), dtype=torch.int32, device=device)
        query_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            *cell_list_cache,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
        )
        assert neighbor_matrix is not None
        assert neighbor_matrix.device == torch.device(device)
        assert neighbor_matrix.dtype == torch.int32
        assert neighbor_matrix.shape == (total_atoms, max_neighbors)
        assert neighbor_matrix_shifts is not None
        assert neighbor_matrix_shifts.device == torch.device(device)
        assert neighbor_matrix_shifts.dtype == torch.int32
        assert neighbor_matrix_shifts.shape == (total_atoms, max_neighbors, 3)
        assert num_neighbors is not None
        assert num_neighbors.device == torch.device(device)
        assert num_neighbors.dtype == torch.int32
        assert num_neighbors.shape == (total_atoms,)

        # Check that we have some neighbors (cubic system should have many)
        valid_neighbors = (neighbor_matrix >= 0).sum()
        assert valid_neighbors > 0

        # Check that the neighbor matrix is correct
        for i in range(total_atoms):
            row_mask = neighbor_matrix[i] >= 0
            assert row_mask.sum() == num_neighbors[i].item()

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_estimate_max_neighbors(self, device, dtype):
        """Test max_neighbors estimation."""
        positions, cell, _ = create_simple_cubic_system(dtype=dtype, device=device)
        cutoff = 1.1
        density = positions.shape[0] / cell.det().abs().item()
        max_neighbors = estimate_max_neighbors(
            cutoff, atomic_density=density, safety_factor=5.0
        )
        assert max_neighbors > 0
        assert isinstance(max_neighbors, int)


class TestTorchCompilability:
    """Test torch.compile compatibility for core functions."""

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_build_cell_list_compile(self, device, dtype):
        """Test that build_cell_list can be compiled with torch.compile."""
        positions, cell, pbc = create_simple_cubic_system(dtype=dtype, device=device)
        cutoff = 1.1
        pbc = pbc.squeeze(0)

        # Get size estimates
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(
            cell,
            pbc,
            cutoff,
        )

        # Test uncompiled version
        clcu = allocate_cell_list(
            positions.shape[0],
            max_cells,
            neighbor_search_radius,
            device,
        )
        build_cell_list(positions, cutoff, cell, pbc, *clcu)

        # Test compiled version
        clcc = allocate_cell_list(
            positions.shape[0],
            max_cells,
            neighbor_search_radius,
            device,
        )

        @torch.compile
        def compiled_build_cell_list(
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
        ):
            build_cell_list(
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
            )

        compiled_build_cell_list(positions, cutoff, cell, pbc, *clcc)

        # Compare results
        for i, (tensor_uncompiled, tensor_compiled) in enumerate(zip(clcu, clcc)):
            assert tensor_uncompiled.shape == tensor_compiled.shape, (
                f"Shape mismatch in tensor {i}: {tensor_uncompiled.shape} vs {tensor_compiled.shape}"
            )
            assert tensor_uncompiled.dtype == tensor_compiled.dtype, (
                f"Dtype mismatch in tensor {i}: {tensor_uncompiled.dtype} vs {tensor_compiled.dtype}"
            )
            assert tensor_uncompiled.device == tensor_compiled.device, (
                f"Device mismatch in tensor {i}: {tensor_uncompiled.device} vs {tensor_compiled.device}"
            )
            # For integer tensors, check exact equality
            if tensor_uncompiled.dtype in [torch.int32, torch.int64]:
                assert torch.equal(tensor_uncompiled, tensor_compiled), (
                    f"Value mismatch in tensor {i}"
                )
            else:
                # For float tensors, use tolerance
                assert torch.allclose(
                    tensor_uncompiled,
                    tensor_compiled,
                    rtol=1e-5,
                    atol=1e-6,
                ), f"Value mismatch in tensor {i}"

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("pbc_flag", [False, True])
    def test_query_cell_list_compile(self, device, dtype, pbc_flag):
        """Test that query_cell_list can be compiled with torch.compile."""
        positions, cell, pbc = create_simple_cubic_system(dtype=dtype, device=device)
        cutoff = 3.0
        pbc = torch.tensor([pbc_flag, pbc_flag, pbc_flag], device=device)
        # Build cell list first
        max_cells, neighbor_search_radius = estimate_cell_list_sizes(
            cell,
            pbc,
            cutoff,
        )
        max_neighbors = estimate_max_neighbors(
            cutoff,
        )
        cell_list_cache_uncompiled = allocate_cell_list(
            positions.shape[0],
            max_cells,
            neighbor_search_radius,
            device,
        )
        build_cell_list(positions, cutoff, cell, pbc, *cell_list_cache_uncompiled)

        # Query cell list
        neighbor_matrix_uncompiled = torch.full(
            (positions.shape[0], max_neighbors),
            fill_value=-1,
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix_shifts_uncompiled = torch.zeros(
            (positions.shape[0], max_neighbors, 3), dtype=torch.int32, device=device
        )
        num_neighbors_uncompiled = torch.zeros(
            (positions.shape[0],), dtype=torch.int32, device=device
        )

        # Test uncompiled version
        query_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            *cell_list_cache_uncompiled,
            neighbor_matrix_uncompiled,
            neighbor_matrix_shifts_uncompiled,
            num_neighbors_uncompiled,
        )

        # Test compiled version
        cell_list_cache_compiled = allocate_cell_list(
            positions.shape[0],
            max_cells,
            neighbor_search_radius.clone(),
            device,
        )
        neighbor_matrix_compiled = torch.full(
            (positions.shape[0], max_neighbors),
            fill_value=-1,
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix_shifts_compiled = torch.zeros(
            (positions.shape[0], max_neighbors, 3), dtype=torch.int32, device=device
        )
        num_neighbors_compiled = torch.zeros(
            (positions.shape[0],), dtype=torch.int32, device=device
        )

        @torch.compile
        def compiled_query_cell_list(
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
        ):
            build_cell_list(
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
            )
            query_cell_list(
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
            )

        compiled_query_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            *cell_list_cache_compiled,
            neighbor_matrix_compiled,
            neighbor_matrix_shifts_compiled,
            num_neighbors_compiled,
        )

        # Compare results
        for row_idx, (unc_row, cmp_row) in enumerate(
            zip(neighbor_matrix_uncompiled, neighbor_matrix_compiled)
        ):
            unc_row_sorted, indices_uncompiled = torch.sort(unc_row)
            cmp_row_sorted, indices_compiled = torch.sort(cmp_row)
            assert torch.equal(unc_row_sorted, cmp_row_sorted), (
                f"Neighbor matrix mismatch for row {row_idx}"
            )
            assert torch.equal(indices_uncompiled, indices_compiled), (
                f"Indices mismatch for row {row_idx}"
            )

            assert torch.equal(
                neighbor_matrix_shifts_uncompiled[row_idx, indices_uncompiled, 0],
                neighbor_matrix_shifts_compiled[row_idx, indices_compiled, 0],
            ), f"Neighbor matrix shifts mismatch for row {row_idx}"
            assert torch.equal(
                neighbor_matrix_shifts_uncompiled[row_idx, indices_uncompiled, 1],
                neighbor_matrix_shifts_compiled[row_idx, indices_compiled, 1],
            ), f"Neighbor matrix shifts mismatch for row {row_idx}"
            assert torch.equal(
                neighbor_matrix_shifts_uncompiled[row_idx, indices_uncompiled, 2],
                neighbor_matrix_shifts_compiled[row_idx, indices_compiled, 2],
            ), f"Neighbor matrix shifts mismatch for row {row_idx}"
        assert torch.equal(num_neighbors_uncompiled, num_neighbors_compiled), (
            "Number of neighbors mismatch"
        )


class TestNaiveDualCutoffMainAPI:
    """Test the main naive dual cutoff neighbor list API function."""

    @pytest.mark.skipif(
        not run_vesin_checks, reason="`vesin` required for consistency checks."
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
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

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
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

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
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


class TestNaiveMainAPI:
    """Test the main naive neighbor list API function."""

    @pytest.mark.skipif(
        not run_vesin_checks, reason="`vesin` required for consistency checks."
    )
    @pytest.mark.parametrize("device", ["cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("half_fill", [False])
    @pytest.mark.parametrize("pbc_flag", [True])
    @pytest.mark.parametrize("preallocate", [True])
    @pytest.mark.parametrize("return_neighbor_list", [True])
    @pytest.mark.parametrize("fill_value", [-1])
    def test_naive_neighbor_matrix_function(
        self,
        device,
        dtype,
        half_fill,
        pbc_flag,
        preallocate,
        return_neighbor_list,
        fill_value,
    ):
        """Test _naive_neighbor_matrix_no_pbc function."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 20

        if not pbc_flag:
            cell = None
            pbc = None
        if preallocate:
            neighbor_matrix = torch.full(
                (positions.shape[0], max_neighbors),
                fill_value,
                dtype=torch.int32,
                device=device,
            )
            num_neighbors = torch.zeros(
                positions.shape[0], dtype=torch.int32, device=device
            )
            args = (positions, cutoff)
            kwargs = {
                "fill_value": fill_value,
                "half_fill": half_fill,
                "neighbor_matrix": neighbor_matrix,
                "num_neighbors": num_neighbors,
                "return_neighbor_list": return_neighbor_list,
            }
            if pbc_flag:
                shift_range_per_dimension, shift_offset, total_shifts = (
                    compute_naive_num_shifts(cell, cutoff, pbc)
                )
                kwargs["cell"] = cell
                kwargs["pbc"] = pbc
                kwargs["shift_range_per_dimension"] = shift_range_per_dimension
                kwargs["shift_offset"] = shift_offset
                kwargs["total_shifts"] = total_shifts
                neighbor_matrix_shifts = torch.zeros(
                    (positions.shape[0], max_neighbors, 3),
                    dtype=torch.int32,
                    device=device,
                )
                kwargs["neighbor_matrix_shifts"] = neighbor_matrix_shifts
            results = naive_neighbor_list(*args, **kwargs)
            if return_neighbor_list:
                if pbc_flag:
                    neighbor_list, neighbor_ptr, neighbor_shifts = results
                    idx_i = neighbor_list[0]
                    idx_j = neighbor_list[1]
                    u = neighbor_shifts
                    num_neighbors = neighbor_ptr[1:] - neighbor_ptr[:-1]
                else:
                    neighbor_list, neighbor_ptr = results
                    idx_i = neighbor_list[0]
                    idx_j = neighbor_list[1]
                    u = torch.zeros(
                        (idx_i.shape[0], 3), dtype=torch.int32, device=device
                    )
                    num_neighbors = neighbor_ptr[1:] - neighbor_ptr[:-1]
        else:
            args = (positions, cutoff)
            kwargs = {
                "max_neighbors": max_neighbors,
                "fill_value": fill_value,
                "half_fill": half_fill,
                "return_neighbor_list": return_neighbor_list,
            }
            if pbc_flag:
                kwargs["cell"] = cell
                kwargs["pbc"] = pbc
            results = naive_neighbor_list(*args, **kwargs)
            if pbc_flag:
                if return_neighbor_list:
                    neighbor_list, neighbor_ptr, neighbor_shifts = results
                    idx_i = neighbor_list[0]
                    idx_j = neighbor_list[1]
                    u = neighbor_shifts
                    num_neighbors = neighbor_ptr[1:] - neighbor_ptr[:-1]
                else:
                    neighbor_matrix, num_neighbors, neighbor_matrix_shifts = results
            else:
                if return_neighbor_list:
                    neighbor_list, neighbor_ptr = results
                    idx_i = neighbor_list[0]
                    idx_j = neighbor_list[1]
                    u = torch.zeros(
                        (idx_i.shape[0], 3), dtype=torch.int32, device=device
                    )
                    num_neighbors = neighbor_ptr[1:] - neighbor_ptr[:-1]
                else:
                    neighbor_matrix, num_neighbors = results

        # Check output shapes and types
        assert num_neighbors.dtype == torch.int32
        assert num_neighbors.shape == (positions.shape[0],)
        assert num_neighbors.device == torch.device(device)
        if return_neighbor_list:
            assert neighbor_list.dtype == torch.int32
            assert neighbor_list.shape == (2, num_neighbors.sum())
            assert neighbor_list.device == torch.device(device)

            if pbc_flag:
                assert u.dtype == torch.int32
                assert u.shape == (num_neighbors.sum(), 3)
                assert u.device == torch.device(device)

        else:
            assert neighbor_matrix.dtype == torch.int32
            assert neighbor_matrix.shape == (
                positions.shape[0],
                max_neighbors,
            )
            assert neighbor_matrix.device == torch.device(device)

            if pbc_flag:
                assert neighbor_matrix_shifts.dtype == torch.int32
                assert neighbor_matrix_shifts.shape == (
                    positions.shape[0],
                    max_neighbors,
                    3,
                )
                assert neighbor_matrix_shifts.device == torch.device(device)

        # Get reference result
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(positions, cell, pbc, cutoff)

        if return_neighbor_list and not half_fill:
            assert_neighbor_lists_equal((idx_i, idx_j, u), (i_ref, j_ref, u_ref))

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_naive_neighbor_list_edge_cases(self, device, dtype, half_fill):
        """Test edge cases for naive_neighbor_list."""
        # Empty system
        positions_empty = torch.empty(0, 3, dtype=dtype, device=device)
        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions=positions_empty,
            cutoff=1.0,
            pbc=None,
            cell=None,
            max_neighbors=10,
            half_fill=half_fill,
        )
        assert neighbor_matrix.shape == (0, 10)
        assert num_neighbors.shape == (0,)

        # Single atom
        positions_single = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions=positions_single,
            cutoff=1.0,
            pbc=None,
            cell=None,
            max_neighbors=10,
            half_fill=half_fill,
        )
        assert num_neighbors[0].item() == 0, "Single atom should have no neighbors"

        # Zero cutoff
        positions, _, _ = create_simple_cubic_system(
            num_atoms=4, dtype=dtype, device=device
        )
        neighbor_matrix, num_neighbors = naive_neighbor_list(
            positions=positions,
            cutoff=0.0,
            pbc=None,
            cell=None,
            max_neighbors=10,
            half_fill=half_fill,
        )
        assert torch.all(num_neighbors == 0), "Zero cutoff should find no neighbors"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_naive_neighbor_list_error_conditions(self, device, dtype, half_fill):
        """Test error conditions for naive_neighbor_list."""
        positions, cell, pbc = create_simple_cubic_system(dtype=dtype, device=device)

        # Test mismatched cell and pbc arguments
        with pytest.raises(
            ValueError, match="If cell is provided, pbc must also be provided"
        ):
            naive_neighbor_list(
                positions,
                1.0,
                pbc=None,
                cell=cell,
                max_neighbors=10,
            )

        with pytest.raises(
            ValueError, match="If pbc is provided, cell must also be provided"
        ):
            naive_neighbor_list(
                positions,
                1.0,
                pbc=pbc,
                cell=None,
                max_neighbors=10,
            )


class TestNaivePerformanceAndScaling:
    """Test performance characteristics and scaling of naive implementation."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_naive_scaling_with_system_size(self, device):
        """Test that naive implementation scales as expected with system size."""
        import time

        dtype = torch.float32
        cutoff = 1.1
        max_neighbors = 100

        # Test different system sizes
        sizes = [10, 50, 100] if device == "cpu" else [50, 100, 200]
        times = []

        for num_atoms in sizes:
            positions, cell, pbc = create_simple_cubic_system(
                num_atoms=num_atoms, dtype=dtype, device=device
            )

            # Warm up
            for _ in range(10):
                naive_neighbor_list(
                    positions,
                    cutoff,
                    pbc=pbc,
                    cell=cell,
                    max_neighbors=max_neighbors,
                )

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            # Time the operation
            start_time = time.time()
            for _ in range(100):
                naive_neighbor_list(
                    positions,
                    cutoff,
                    pbc=pbc,
                    cell=cell,
                    max_neighbors=max_neighbors,
                )

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            elapsed = time.time() - start_time
            times.append(elapsed)

        # Check that it doesn't grow too fast (should be roughly O(N^2))
        # This is a loose check since we can't expect perfect scaling
        assert times[1] > times[0] * 0.8, "Time should increase with system size"
        if len(times) > 2:
            # Very loose scaling check
            scaling_factor = times[-1] / times[0]
            size_factor = (sizes[-1] / sizes[0]) ** 2
            assert scaling_factor < size_factor * 5, (
                "Scaling should not be much worse than O(N^2)"
            )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_naive_cutoff_scaling(self, device):
        """Test scaling with different cutoff values."""
        dtype = torch.float32
        num_atoms = 50
        max_neighbors = 200

        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=num_atoms, dtype=dtype, device=device
        )

        # Test different cutoffs
        cutoffs = [0.5, 1.0, 1.5, 2.0]
        neighbor_counts = []

        for cutoff in cutoffs:
            _, num_neighbors, _ = naive_neighbor_list(
                positions,
                cutoff,
                pbc=pbc,
                cell=cell,
                max_neighbors=max_neighbors,
            )
            total_pairs = num_neighbors.sum().item()
            neighbor_counts.append(total_pairs)

        # Check that neighbor count increases with cutoff
        for i in range(1, len(neighbor_counts)):
            assert neighbor_counts[i] >= neighbor_counts[i - 1], (
                f"Neighbor count should increase with cutoff: {neighbor_counts}"
            )


class TestNaiveRobustness:
    """Test robustness of naive implementation to various inputs."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_random_systems(self, device, dtype, half_fill):
        """Test with random systems of various sizes and configurations."""
        for pbc_flag in [True, False]:
            # Test several random systems
            for seed in [42, 123, 456]:
                positions, cell, pbc = create_random_system(
                    num_atoms=20,
                    cell_size=3.0,
                    dtype=dtype,
                    device=device,
                    seed=seed,
                    pbc_flag=pbc_flag,
                )
                cutoff = 1.2
                max_neighbors = 50

                # Should not crash
                neighbor_matrix, num_neighbors, unit_shifts = naive_neighbor_list(
                    positions=positions,
                    cutoff=cutoff,
                    pbc=pbc,
                    cell=cell,
                    max_neighbors=max_neighbors,
                    half_fill=half_fill,
                )

                # Basic sanity checks
                assert torch.all(num_neighbors >= 0)
                assert torch.all(num_neighbors <= max_neighbors)
                assert neighbor_matrix.device == torch.device(device)
                assert unit_shifts.device == torch.device(device)
                assert num_neighbors.device == torch.device(device)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_extreme_geometries(self, device, dtype, half_fill):
        """Test with extreme cell geometries."""
        # Very elongated cell
        positions = torch.rand(10, 3, dtype=dtype, device=device)
        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]],
            dtype=dtype,
            device=device,
        ).reshape(1, 3, 3)
        pbc = torch.tensor([True, True, True], device=device).reshape(1, 3)
        cutoff = 0.2
        max_neighbors = 20

        # Should handle extreme aspect ratios
        _, num_neighbors, _ = naive_neighbor_list(
            positions=positions * torch.tensor([10.0, 0.1, 0.1], device=device),
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
        )

        assert torch.all(num_neighbors >= 0)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_large_cutoffs(self, device, dtype, half_fill):
        """Test with very large cutoffs."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )

        # Cutoff larger than cell size
        large_cutoff = 5.0
        max_neighbors = 200

        _, num_neighbors, _ = naive_neighbor_list(
            positions=positions,
            cutoff=large_cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
        )

        # Should find many neighbors
        assert num_neighbors.sum() > 0
        # Each atom should have multiple neighbors (including periodic images)
        assert torch.all(num_neighbors > 0)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_precision_consistency(self, device, dtype, half_fill):
        """Test that float32 and float64 give consistent results."""
        positions_f32, cell_f32, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=torch.float32, device=device
        )
        positions_f64 = positions_f32.double()
        cell_f64 = cell_f32.double()

        cutoff = 1.1
        max_neighbors = 50

        # Get results for both precisions
        _, num_neighbors_f32, _ = naive_neighbor_list(
            positions_f32,
            cutoff,
            pbc=pbc,
            cell=cell_f32,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
        )
        _, num_neighbors_f64, _ = naive_neighbor_list(
            positions_f64,
            cutoff,
            pbc=pbc,
            cell=cell_f64,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
        )

        # Neighbor counts should be identical (for this exact geometry)
        torch.testing.assert_close(num_neighbors_f32, num_neighbors_f64, rtol=0, atol=0)


class TestNaiveTorchCompilability:
    """Test torch.compile compatibility for naive neighbor list functions."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [False, True])
    def test_naive_neighbor_list_compile_no_pbc(self, device, dtype, half_fill):
        """Test that naive_neighbor_list can be compiled with torch.compile."""
        positions, _, _ = create_simple_cubic_system(
            num_atoms=50, dtype=dtype, device=device
        )
        cutoff = 3.0
        max_neighbors = 100

        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors),
            50,
            dtype=torch.int32,
            device=device,
        )
        num_neighbors = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )

        # Test compiled version
        @torch.compile
        def compiled_naive_neighbor_list(
            positions,
            cutoff,
            neighbor_matrix,
            num_neighbors,
            half_fill,
        ):
            return naive_neighbor_list(
                positions=positions,
                cutoff=cutoff,
                neighbor_matrix=neighbor_matrix,
                num_neighbors=num_neighbors,
                half_fill=half_fill,
            )

        compiled_naive_neighbor_list(
            positions,
            cutoff,
            neighbor_matrix,
            num_neighbors,
            half_fill,
        )

        assert num_neighbors.sum() > 0
        num_rows = positions.shape[0] - int(half_fill)
        for i in range(num_rows):
            assert num_neighbors[i].item() > 0
            neighbor_row = neighbor_matrix[i]
            mask = neighbor_row != 50
            assert neighbor_row[mask].shape == (num_neighbors[i].item(),)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_naive_neighbor_list_compile_pbc(self, device, dtype, half_fill):
        """Test that naive_neighbor_list can be compiled with torch.compile."""
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=50, dtype=dtype, device=device
        )
        cutoff = 1.1
        max_neighbors = 50
        cell = cell.reshape(1, 3, 3)
        pbc = pbc.reshape(1, 3)
        shift_range_per_dimension, shift_offset, total_shifts = (
            compute_naive_num_shifts(cell, cutoff, pbc)
        )

        neighbor_matrix = torch.full(
            (positions.shape[0], max_neighbors),
            50,
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix_shifts = torch.zeros(
            (positions.shape[0], max_neighbors, 3),
            dtype=torch.int32,
            device=device,
        )
        num_neighbors = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=device
        )

        # Test compiled version
        @torch.compile
        def compiled_naive_neighbor_list(
            positions,
            cutoff,
            cell,
            pbc,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            shift_range_per_dimension,
            shift_offset,
            total_shifts,
            half_fill,
        ):
            return naive_neighbor_list(
                positions=positions,
                cutoff=cutoff,
                cell=cell,
                pbc=pbc,
                neighbor_matrix=neighbor_matrix,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                num_neighbors=num_neighbors,
                shift_range_per_dimension=shift_range_per_dimension,
                shift_offset=shift_offset,
                total_shifts=total_shifts,
                half_fill=half_fill,
            )

        compiled_naive_neighbor_list(
            positions,
            cutoff,
            cell,
            pbc,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            shift_range_per_dimension,
            shift_offset,
            total_shifts,
            half_fill,
        )

        # Compare results
        assert num_neighbors.sum() > 0
        num_rows = positions.shape[0] - int(half_fill)
        for i in range(num_rows):
            assert num_neighbors[i].item() > 0
            neighbor_row = neighbor_matrix[i]
            mask = neighbor_row != 50
            assert neighbor_row[mask].shape == (num_neighbors[i].item(),)


class TestNaiveMemoryAndPerformance:
    """Test memory usage and performance characteristics of naive implementation."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_memory_scaling(self, device, half_fill):
        """Test that memory usage scales reasonably with system size."""
        import gc

        dtype = torch.float32
        cutoff = 1.1

        # Test different system sizes
        sizes = [10, 20] if device == "cpu" else [50, 100]

        for num_atoms in sizes:
            positions, cell, pbc = create_simple_cubic_system(
                num_atoms=num_atoms, dtype=dtype, device=device
            )
            cell = cell.reshape(1, 3, 3)
            pbc = pbc.reshape(1, 3)

            # Estimate reasonable max_neighbors based on system size and cutoff
            max_neighbors = 100

            # Clear cache before test
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

            # Run batch naive implementation
            neighbor_matrix, num_neighbors, unit_shifts = naive_neighbor_list(
                positions=positions,
                cutoff=cutoff,
                pbc=pbc,
                cell=cell,
                max_neighbors=max_neighbors,
                half_fill=half_fill,
            )

            # Basic checks that output is reasonable
            assert neighbor_matrix.shape == (
                num_atoms,
                max_neighbors,
            )
            assert unit_shifts.shape == (num_atoms, max_neighbors, 3)
            assert num_neighbors.shape == (num_atoms,)
            assert torch.all(num_neighbors >= 0)
            assert torch.all(num_neighbors <= max_neighbors)

            # Clean up
            del neighbor_matrix, unit_shifts, num_neighbors, positions, cell, pbc
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_max_neighbors_overflow_handling(self, device, dtype, half_fill):
        """Test behavior when max_neighbors is exceeded."""

        # Create a dense system with small max_neighbors to force overflow
        positions, cell, pbc = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        cell = cell.reshape(1, 3, 3)
        pbc = pbc.reshape(1, 3)

        cutoff = 2.0  # Large cutoff to find many neighbors
        max_neighbors = 3  # Artificially small to trigger overflow

        # Should not crash, but may not find all neighbors
        neighbor_matrix, num_neighbors, unit_shifts = naive_neighbor_list(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            cell=cell,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
        )

        # Should still produce valid output, just potentially incomplete
        assert torch.all(num_neighbors >= 0)
        assert neighbor_matrix.shape == (positions.shape[0], max_neighbors)
        assert unit_shifts.shape == (positions.shape[0], max_neighbors, 3)
        assert num_neighbors.shape == (positions.shape[0],)
        assert neighbor_matrix.device == torch.device(device)
        assert unit_shifts.device == torch.device(device)
        assert num_neighbors.device == torch.device(device)
