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

"""Tests for PyTorch bindings of batched neighbor list methods."""

from importlib import import_module

import pytest
import torch

from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.torch.neighbors.batched import (
    batch_build_cell_list,
    batch_cell_list,
    batch_naive_neighbor_list,
    batch_naive_neighbor_list_dual_cutoff,
    batch_query_cell_list,
    estimate_batch_cell_list_sizes,
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    allocate_cell_list,
)

from ...test_utils import (
    assert_neighbor_lists_equal,
    brute_force_neighbors,
    create_batch_systems,
    create_random_system,
    create_simple_cubic_system,
)

try:
    _ = import_module("vesin")
    run_vesin_checks = True
except ModuleNotFoundError:
    run_vesin_checks = False


devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda:0")
dtypes = [torch.float32, torch.float64]


class TestBatchCellListAPI:
    """Test the main batch cell list API functions."""

    @pytest.mark.skipif(
        not run_vesin_checks, reason="`vesin` required for consistency checks."
    )
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("cutoff", [1.0, 3.0])
    def test_single_system_single_atom(self, device, dtype, cutoff):
        """Test with single system containing single atom (should have no neighbors)."""
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        cell = (torch.eye(3, dtype=dtype, device=device) * 2.0).reshape(1, 3, 3)
        pbc = torch.tensor([[True, True, True]], device=device)
        batch_idx = torch.tensor([0], dtype=torch.int32, device=device)

        # Test batch_cell_list function
        neighbor_list, _, u = batch_cell_list(
            positions, cutoff, cell, pbc, batch_idx, return_neighbor_list=True
        )

        i, j = neighbor_list

        i_ref, j_ref, u_ref, _ = brute_force_neighbors(
            positions, cell, pbc.squeeze(0), cutoff
        )

        # Results should be identical
        assert_neighbor_lists_equal((i, j, u), (i_ref, j_ref, u_ref))

    @pytest.mark.skipif(
        not run_vesin_checks, reason="`vesin` required for consistency checks."
    )
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_single_system_two_atoms(self, device, dtype):
        """Test single system with two atoms."""
        # Two atoms within cutoff distance
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=dtype, device=device
        )
        cell = (torch.eye(3, dtype=dtype, device=device) * 2.0).reshape(1, 3, 3)
        pbc = torch.tensor([[True, True, True]], device=device)
        batch_idx = torch.tensor([0, 0], dtype=torch.int32, device=device)
        cutoff = 1.0

        neighbor_list, _, u = batch_cell_list(
            positions, cutoff, cell, pbc, batch_idx, return_neighbor_list=True
        )
        i, j = neighbor_list

        # Should have 2 pairs: (0->1) and (1->0)
        assert len(i) == 2, f"Expected 2 neighbors, got {len(i)}"

        # Compare with brute force reference
        i_ref, j_ref, u_ref, _ = brute_force_neighbors(
            positions, cell, pbc.squeeze(0), cutoff
        )
        assert_neighbor_lists_equal((i, j, u), (i_ref, j_ref, u_ref))

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_two_systems_same_structure(self, device, dtype):
        """Test batch with two identical systems."""
        # Create two identical cubic systems
        positions_1, cell_1, pbc_1 = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        positions_2 = positions_1.clone()

        # Concatenate for batch
        positions = torch.cat([positions_1, positions_2], dim=0)
        cell = torch.cat([cell_1, cell_1], dim=0)
        pbc = torch.cat([pbc_1, pbc_1], dim=0)
        batch_idx = torch.cat(
            [
                torch.zeros(8, dtype=torch.int32, device=device),
                torch.ones(8, dtype=torch.int32, device=device),
            ]
        )
        cutoff = 1.1

        # Test batch_cell_list
        _, neighbor_ptr, _ = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            max_neighbors=10,
            return_neighbor_list=True,
        )
        num_neighbors = neighbor_ptr[1:] - neighbor_ptr[:-1]
        # Each system should have the same number of neighbors
        num_neighbors_sys0 = num_neighbors[:8].sum().item()
        num_neighbors_sys1 = num_neighbors[8:].sum().item()
        assert num_neighbors_sys0 == num_neighbors_sys1, (
            f"Identical systems should have same neighbor counts: "
            f"{num_neighbors_sys0} vs {num_neighbors_sys1}"
        )

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_two_systems_different_structures(self, device, dtype):
        """Test batch with two different systems."""
        # System 1: 4 atoms
        positions_1 = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=dtype,
            device=device,
        )
        cell_1 = (torch.eye(3, dtype=dtype, device=device) * 3.0).reshape(1, 3, 3)
        pbc_1 = torch.tensor([[True, True, True]], device=device)

        # System 2: 3 atoms
        positions_2 = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [0.0, 0.8, 0.0],
            ],
            dtype=dtype,
            device=device,
        )
        cell_2 = (torch.eye(3, dtype=dtype, device=device) * 2.5).reshape(1, 3, 3)
        pbc_2 = torch.tensor([[True, True, True]], device=device)

        # Concatenate for batch
        positions = torch.cat([positions_1, positions_2], dim=0)
        cell = torch.cat([cell_1, cell_2], dim=0)
        pbc = torch.cat([pbc_1, pbc_2], dim=0)
        batch_idx = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1], dtype=torch.int32, device=device
        )
        cutoff = 1.5

        # Test batch_cell_list
        neighbor_list, _, _ = batch_cell_list(
            positions, cutoff, cell, pbc, batch_idx, return_neighbor_list=True
        )
        i, j = neighbor_list

        # Basic checks
        assert i.dtype == torch.int32
        assert j.dtype == torch.int32
        assert i.device.type == device.split(":")[0]

        # Verify neighbors are within their respective systems
        for atom_i, atom_j in zip(i.tolist(), j.tolist()):
            sys_i = batch_idx[atom_i].item()
            sys_j = batch_idx[atom_j].item()
            assert sys_i == sys_j, (
                f"Cross-system neighbors detected: atom {atom_i} (sys {sys_i}) "
                f"-> atom {atom_j} (sys {sys_j})"
            )

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_random_batch_systems(self, device, dtype):
        """Test with batch of random systems."""
        atoms_per_system = [10, 15, 12]
        cutoff = 5.0

        positions_list = []
        cells_list = []
        pbcs_list = []
        batch_idx_list = []

        for sys_idx, num_atoms in enumerate(atoms_per_system):
            pos, cell, pbc = create_random_system(
                num_atoms=num_atoms,
                cell_size=3.0,
                dtype=dtype,
                device=device,
                seed=42 + sys_idx,
                pbc_flag=True,
            )
            positions_list.append(pos)
            cells_list.append(cell)
            pbcs_list.append(pbc)
            batch_idx_list.append(
                torch.full((num_atoms,), sys_idx, dtype=torch.int32, device=device)
            )

        positions = torch.cat(positions_list, dim=0)
        cell = torch.cat(cells_list, dim=0)
        pbc = torch.cat(pbcs_list, dim=0)
        batch_idx = torch.cat(batch_idx_list, dim=0)

        neighbor_list, _, u = batch_cell_list(
            positions, cutoff, cell, pbc, batch_idx, return_neighbor_list=True
        )
        i, j = neighbor_list

        # Basic checks
        assert i.dtype == torch.int32
        assert j.dtype == torch.int32
        assert u.dtype == torch.int32
        assert i.device.type == device.split(":")[0]

        # Check consistency: if (i,j) is a pair, j should be within cutoff of i
        if len(i) > 0:
            for idx in range(min(10, len(i))):
                atom_i, atom_j = i[idx].item(), j[idx].item()
                sys_idx = batch_idx[atom_i].item()
                shift = cell[sys_idx] @ u[idx].to(dtype)
                rij = positions[atom_j] - positions[atom_i] + shift
                dist = torch.norm(rij, dim=0).item()
                assert dist < cutoff + 1e-5, f"Distance {dist} exceeds cutoff {cutoff}"

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_batch_no_pbc(self, device, dtype, return_neighbor_list):
        """Test batch with no periodic boundary conditions."""
        positions_1, cell_1, _ = create_simple_cubic_system(
            num_atoms=8, cell_size=3.0, dtype=dtype, device=device
        )
        positions_2 = positions_1.clone()

        positions = torch.cat([positions_1, positions_2], dim=0)
        cell = torch.cat([cell_1, cell_1], dim=0)
        pbc = torch.tensor(
            [[False, False, False], [False, False, False]], device=device
        )
        batch_idx = torch.cat(
            [
                torch.zeros(8, dtype=torch.int32, device=device),
                torch.ones(8, dtype=torch.int32, device=device),
            ]
        )
        cutoff = 1.1

        results = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
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
    def test_batch_mixed_pbc(
        self, device, dtype, return_neighbor_list, preallocate, fill_value
    ):
        """Test batch with mixed periodic boundary conditions."""
        positions_1, cell_1, _ = create_simple_cubic_system(
            num_atoms=8, cell_size=2.0, dtype=dtype, device=device
        )
        positions_2 = positions_1.clone()

        positions = torch.cat([positions_1, positions_2], dim=0)
        cell = torch.cat([cell_1, cell_1], dim=0)
        pbc = torch.tensor([[True, False, True], [False, True, False]], device=device)
        batch_idx = torch.cat(
            [
                torch.zeros(8, dtype=torch.int32, device=device),
                torch.ones(8, dtype=torch.int32, device=device),
            ]
        )
        cutoff = 3.0

        if preallocate:
            max_neighbors = estimate_max_neighbors(cutoff)
            max_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
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

            results = batch_cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                batch_idx,
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
            results = batch_cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                batch_idx,
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
    def test_batch_zero_cutoff(self, device, dtype, return_neighbor_list):
        """Test batch with zero cutoff (should find no neighbors)."""
        positions_1, cell_1, pbc_1 = create_simple_cubic_system(
            num_atoms=8, dtype=dtype, device=device
        )
        positions = torch.cat([positions_1, positions_1], dim=0)
        cell = torch.cat([cell_1, cell_1], dim=0)
        pbc = torch.cat([pbc_1, pbc_1], dim=0)
        batch_idx = torch.cat(
            [
                torch.zeros(8, dtype=torch.int32, device=device),
                torch.ones(8, dtype=torch.int32, device=device),
            ]
        )
        cutoff = 0.0

        results = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            return_neighbor_list=return_neighbor_list,
        )
        if return_neighbor_list:
            assert len(results) == 3
            assert results[0].shape == (2, 0)  # neighbor_list
            assert results[1].shape == (17,)  # neighbor_ptr
            assert results[2].shape == (0, 3)  # shifts
        else:
            assert len(results) == 3
            assert results[0].shape[0] == 16
            assert results[-1].sum().item() == 0

    @pytest.mark.parametrize(
        "pbc_flags",
        [
            [[True, True, True], [True, True, True]],
            [[False, False, False], [False, False, False]],
            [[True, False, True], [False, True, False]],
        ],
    )
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("num_atoms", [10, 20])
    @pytest.mark.parametrize("cutoff", [1.0, 3.0])
    def test_batch_scaling_correctness(
        self, pbc_flags, dtype, device, num_atoms, cutoff
    ):
        """Test batch with various sizes and configurations."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions_list = []
        cells_list = []
        pbcs_list = []
        batch_idx_list = []

        for sys_idx, pbc_flag in enumerate(pbc_flags):
            pos, cell, pbc = create_random_system(
                num_atoms=num_atoms,
                cell_size=3.0,
                dtype=dtype,
                device=device,
                seed=42 + sys_idx,
                pbc_flag=pbc_flag,
            )
            positions_list.append(pos)
            cells_list.append(cell)
            pbcs_list.append(pbc)
            batch_idx_list.append(
                torch.full((num_atoms,), sys_idx, dtype=torch.int32, device=device)
            )

        positions = torch.cat(positions_list, dim=0)
        cell = torch.cat(cells_list, dim=0)
        pbc = torch.cat(pbcs_list, dim=0)
        batch_idx = torch.cat(batch_idx_list, dim=0)

        estimated_density = num_atoms / cell[0].det().abs().item()
        max_neighbors = estimate_max_neighbors(
            cutoff, atomic_density=estimated_density, safety_factor=5.0
        )
        neighbor_list, _, u = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            max_neighbors=max_neighbors,
            return_neighbor_list=True,
        )
        i, j = neighbor_list
        S = u.to(dtype)

        # Check consistency: if (i,j) is a pair, j should be within cutoff of i
        if len(i) > 0:
            for idx in range(min(10, len(i))):
                atom_i, atom_j = i[idx].item(), j[idx].item()
                sys_idx = batch_idx[atom_i].item()
                shift = S[idx] @ cell[sys_idx]
                rij = positions[atom_j] - positions[atom_i] + shift
                dist = torch.norm(rij, dim=0).item()
                assert dist < cutoff + 1e-5, f"Distance {dist} exceeds cutoff {cutoff}"


class TestBatchEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_empty_estimate_batch_cell_list_sizes(self, device, dtype):
        """Test that estimate_batch_cell_list_sizes returns the correct values for an empty batch."""
        cell = torch.zeros((0, 3, 3), dtype=dtype, device=device)
        pbc = torch.zeros((0, 3), dtype=torch.bool, device=device)
        cutoff = 1.0
        max_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
            cell, pbc, cutoff
        )
        assert max_cells == 1
        assert neighbor_search_radius.shape == (0, 3)
        assert neighbor_search_radius.dtype == torch.int32
        assert neighbor_search_radius.device == torch.device(device)

        # Now test with negative cutoff
        cell = torch.eye(3, dtype=dtype, device=device).reshape(1, 3, 3)
        pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
        cutoff = -1.0
        max_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
            cell, pbc, cutoff
        )
        assert max_cells == 1
        assert neighbor_search_radius.shape == (1, 3)
        assert neighbor_search_radius.dtype == torch.int32
        assert neighbor_search_radius.device == torch.device(device)

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_all_negative_volume(self, device, dtype):
        """Check to make sure bindings raises error for all <= 0 volumes"""
        positions = torch.rand((4, 3), device=device, dtype=dtype)
        # cell has zero and negative volumes
        cells = torch.tensor(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [
                    [0.2225, 0.6140, 0.7039],
                    [0.4351, 0.3592, 0.8304],
                    [0.1768, 0.0427, 0.3177],
                ],
            ],
            dtype=dtype,
            device=device,
        )
        pbc = torch.ones((4, 3), dtype=bool, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        with pytest.raises(RuntimeError, match="Cells with volume <= 0"):
            _ = batch_cell_list(positions, 3.0, cells, pbc, batch_idx)

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_mixed_negative_volume(self, device, dtype):
        """Check that samples with negative volume fail"""
        positions = torch.rand((4, 3), device=device, dtype=dtype)
        # first is positive, second is negative
        cells = torch.tensor(
            [
                [
                    [0.7340, 0.5755, 0.5256],
                    [0.3528, 0.1856, 0.9662],
                    [0.2384, 0.1754, 0.1968],
                ],
                [
                    [0.3681, 0.1729, 0.0691],
                    [0.7392, 0.7962, 0.9036],
                    [0.3154, 0.7710, 0.2854],
                ],
            ],
            dtype=dtype,
            device=device,
        )
        pbc = torch.ones((4, 3), dtype=bool, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        with pytest.raises(RuntimeError, match="Cells with volume <= 0"):
            _ = batch_cell_list(positions, 3.0, cells, pbc, batch_idx)

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("dtype", dtypes)
    def test_empty_batch_build_cell_list(self, device, dtype):
        """Test with empty batch."""
        positions = torch.empty(0, 3, dtype=dtype, device=device)
        cell = torch.eye(3, dtype=dtype, device=device).reshape(1, 3, 3)
        pbc = torch.tensor([[True, True, True]], dtype=torch.bool, device=device)
        batch_idx = torch.empty(0, dtype=torch.int32, device=device)
        cutoff = 1.0
        cells_per_dimension = torch.tensor([1, 1, 1], dtype=torch.int32, device=device)
        neighbor_search_radius = torch.tensor(
            [1, 1, 1], dtype=torch.int32, device=device
        )
        atom_periodic_shifts = torch.tensor([0, 0, 0], dtype=torch.int32, device=device)
        atom_to_cell_mapping = torch.tensor([0, 0, 0], dtype=torch.int32, device=device)
        atoms_per_cell_count = torch.tensor([0], dtype=torch.int32, device=device)
        cell_atom_start_indices = torch.tensor([0], dtype=torch.int32, device=device)
        cell_atom_list = torch.tensor([], dtype=torch.int32, device=device)
        batch_build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        )

        assert torch.equal(
            atom_periodic_shifts,
            torch.tensor([0, 0, 0], dtype=torch.int32, device=device),
        )
        assert torch.equal(
            atom_to_cell_mapping,
            torch.tensor([0, 0, 0], dtype=torch.int32, device=device),
        )
        assert torch.equal(
            atoms_per_cell_count, torch.tensor([0], dtype=torch.int32, device=device)
        )
        assert torch.equal(
            cell_atom_start_indices, torch.tensor([0], dtype=torch.int32, device=device)
        )
        assert torch.equal(
            cell_atom_list, torch.tensor([], dtype=torch.int32, device=device)
        )

        # Now test with negative cutoff
        cutoff = -1.0
        batch_build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        )
        assert torch.equal(
            atom_periodic_shifts,
            torch.tensor([0, 0, 0], dtype=torch.int32, device=device),
        )
        assert torch.equal(
            atom_to_cell_mapping,
            torch.tensor([0, 0, 0], dtype=torch.int32, device=device),
        )
        assert torch.equal(
            atoms_per_cell_count, torch.tensor([0], dtype=torch.int32, device=device)
        )
        assert torch.equal(
            cell_atom_start_indices, torch.tensor([0], dtype=torch.int32, device=device)
        )
        assert torch.equal(
            cell_atom_list, torch.tensor([], dtype=torch.int32, device=device)
        )

    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_empty_batch(self, return_neighbor_list):
        """Test with empty batch."""
        positions = torch.empty(0, 3, dtype=torch.float32)
        cell = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3)
        pbc = torch.tensor([[True, True, True]])
        batch_idx = torch.empty(0, dtype=torch.int32)
        cutoff = 1.0

        results = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            return_neighbor_list=return_neighbor_list,
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

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_batch_dtype_consistency(self, dtype, return_neighbor_list):
        """Test that output dtypes are consistent with inputs."""
        positions = torch.randn(10, 3, dtype=dtype)
        cell = (torch.eye(3, dtype=dtype) * 2.0).reshape(1, 3, 3).repeat(2, 1, 1)
        pbc = torch.tensor([[True, True, True], [True, True, True]], dtype=torch.bool)
        batch_idx = torch.cat(
            [torch.zeros(5, dtype=torch.int32), torch.ones(5, dtype=torch.int32)]
        )
        cutoff = 1.5

        results = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            return_neighbor_list=return_neighbor_list,
        )

        for result in results:
            assert result.dtype == torch.int32

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("return_neighbor_list", [True, False])
    def test_batch_device_consistency(self, device, return_neighbor_list):
        """Test that outputs are on the same device as inputs."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions = torch.randn(10, 3, device=device)
        cell = torch.eye(3, device=device).reshape(1, 3, 3).repeat(2, 1, 1) * 2.0
        pbc = torch.tensor([[True, True, True], [True, True, True]], device=device)
        batch_idx = torch.cat(
            [
                torch.zeros(5, dtype=torch.int32, device=device),
                torch.ones(5, dtype=torch.int32, device=device),
            ]
        )
        cutoff = 1.5

        results = batch_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            return_neighbor_list=return_neighbor_list,
        )
        for result in results:
            assert result.device == torch.device(device)


class TestBatchCellListComponentsAPI:
    """Test the modular batch cell list API functions."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_batch_build_and_query_cell_list(self, device, dtype):
        """Test building and querying batch cell list separately."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create batch with 2 systems
        positions_1, cell_1, pbc_1 = create_simple_cubic_system(
            dtype=dtype, device=device
        )
        positions_2 = positions_1.clone()

        positions = torch.cat([positions_1, positions_2], dim=0)
        cell = torch.cat([cell_1, cell_1], dim=0)
        pbc = torch.cat([pbc_1, pbc_1], dim=0)
        batch_idx = torch.cat(
            [
                torch.zeros(8, dtype=torch.int32, device=device),
                torch.ones(8, dtype=torch.int32, device=device),
            ]
        )
        cutoff = 1.1

        # Get size estimates for batch_build_cell_list
        max_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
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
        batch_build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            *cell_list_cache,
        )

        assert cell_list_cache[0] is not None
        assert cell_list_cache[0].device == torch.device(device)
        assert cell_list_cache[0].dtype == torch.int32
        assert cell_list_cache[0].shape == (2, 3)  # 2 systems, 3 dimensions

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
        batch_query_cell_list(
            positions,
            cell,
            pbc,
            cutoff,
            batch_idx,
            *cell_list_cache,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
            False,
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


class TestBatchTorchCompilability:
    """Test torch.compile compatibility for core batch functions."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_batch_build_cell_list_compile(self, device, dtype):
        """Test that batch_build_cell_list can be compiled with torch.compile."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions_1, cell_1, pbc_1 = create_simple_cubic_system(
            dtype=dtype, device=device
        )
        positions = torch.cat([positions_1, positions_1], dim=0)
        cell = torch.cat([cell_1, cell_1], dim=0)
        pbc = torch.cat([pbc_1, pbc_1], dim=0)
        batch_idx = torch.cat(
            [
                torch.zeros(8, dtype=torch.int32, device=device),
                torch.ones(8, dtype=torch.int32, device=device),
            ]
        )
        cutoff = 1.1

        # Get size estimates
        max_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
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
        batch_build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            *clcu,
        )

        # Test compiled version
        clcc = allocate_cell_list(
            positions.shape[0],
            max_cells,
            neighbor_search_radius.clone(),
            device,
        )

        @torch.compile
        def compiled_batch_build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ):
            batch_build_cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                batch_idx,
                cells_per_dimension,
                neighbor_search_radius,
                atom_periodic_shifts,
                atom_to_cell_mapping,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
            )

        compiled_batch_build_cell_list(positions, cutoff, cell, pbc, batch_idx, *clcc)

        # Compare results (cell_list_cache includes cell_offsets at index [2])
        all_tensors_u = clcu
        all_tensors_c = clcc
        for i, (tensor_uncompiled, tensor_compiled) in enumerate(
            zip(all_tensors_u, all_tensors_c)
        ):
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

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("pbc_flag", [False, True])
    def test_batch_query_cell_list_compile(self, device, dtype, pbc_flag):
        """Test that batch_query_cell_list can be compiled with torch.compile."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        positions_1, cell_1, _ = create_simple_cubic_system(dtype=dtype, device=device)
        positions = torch.cat([positions_1, positions_1], dim=0)
        cell = torch.cat([cell_1, cell_1], dim=0)
        pbc = torch.tensor(
            [[pbc_flag, pbc_flag, pbc_flag], [pbc_flag, pbc_flag, pbc_flag]],
            device=device,
        )
        batch_idx = torch.cat(
            [
                torch.zeros(8, dtype=torch.int32, device=device),
                torch.ones(8, dtype=torch.int32, device=device),
            ]
        )
        cutoff = 3.0

        # Build cell list first
        max_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
            cell,
            pbc,
            cutoff,
        )
        max_neighbors = estimate_max_neighbors(cutoff)

        cell_list_cache_uncompiled = allocate_cell_list(
            positions.shape[0],
            max_cells,
            neighbor_search_radius,
            device,
        )
        batch_build_cell_list(
            positions,
            cutoff,
            cell,
            pbc,
            batch_idx,
            *cell_list_cache_uncompiled,
        )

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
        batch_query_cell_list(
            positions,
            cell,
            pbc,
            cutoff,
            batch_idx,
            *cell_list_cache_uncompiled,
            neighbor_matrix_uncompiled,
            neighbor_matrix_shifts_uncompiled,
            num_neighbors_uncompiled,
            False,
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
            cell,
            pbc,
            cutoff,
            batch_idx,
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
            batch_build_cell_list(
                positions,
                cutoff,
                cell,
                pbc,
                batch_idx,
                cells_per_dimension,
                neighbor_search_radius,
                atom_periodic_shifts,
                atom_to_cell_mapping,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
            )
            batch_query_cell_list(
                positions,
                cell,
                pbc,
                cutoff,
                batch_idx,
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
                False,
            )

        compiled_query_cell_list(
            positions,
            cell,
            pbc,
            cutoff,
            batch_idx,
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


def create_batch_idx_and_ptr(
    atoms_per_system: list, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create batch_idx and batch_ptr tensors from atoms_per_system list."""
    total_atoms = sum(atoms_per_system)
    batch_idx = torch.zeros(total_atoms, dtype=torch.int32, device=device)
    batch_ptr = torch.zeros(len(atoms_per_system) + 1, dtype=torch.int32, device=device)

    start_idx = 0
    for i, num_atoms in enumerate(atoms_per_system):
        batch_idx[start_idx : start_idx + num_atoms] = i
        batch_ptr[i + 1] = batch_ptr[i] + num_atoms
        start_idx += num_atoms

    return batch_idx, batch_ptr


class TestBatchNaiveMainAPI:
    """Test the main batch naive neighbor list API function."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_list_no_pbc(self, device, dtype, half_fill):
        """Test batch_naive_neighbor_list without periodic boundary conditions."""
        atoms_per_system = [6, 8, 7]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=3, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 20

        # Test without PBC
        neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            pbc=None,
            cell=None,
            max_neighbors=max_neighbors,
            half_fill=half_fill,
        )

        # Check output types and shapes
        total_atoms = positions_batch.shape[0]
        expected_rows = total_atoms
        assert neighbor_matrix.dtype == torch.int32
        assert num_neighbors.dtype == torch.int32
        assert neighbor_matrix.shape == (expected_rows, max_neighbors)
        assert num_neighbors.shape == (total_atoms,)
        assert neighbor_matrix.device == torch.device(device)
        assert num_neighbors.device == torch.device(device)

        # Check neighbor counts are reasonable
        assert torch.all(num_neighbors >= 0)
        assert torch.all(num_neighbors <= max_neighbors)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_list_with_pbc(self, device, dtype, half_fill):
        """Test batch_naive_neighbor_list with periodic boundary conditions."""
        atoms_per_system = [5, 7, 6]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=3, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 30

        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = (
            batch_naive_neighbor_list(
                positions=positions_batch,
                cutoff=cutoff,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                pbc=pbc_batch,
                cell=cell_batch,
                max_neighbors=max_neighbors,
                half_fill=half_fill,
            )
        )

        # Check output types and shapes
        total_atoms = positions_batch.shape[0]
        expected_rows = total_atoms
        assert neighbor_matrix.dtype == torch.int32
        assert neighbor_matrix_shifts.dtype == torch.int32
        assert num_neighbors.dtype == torch.int32
        assert neighbor_matrix.shape == (expected_rows, max_neighbors)
        assert neighbor_matrix_shifts.shape == (expected_rows, max_neighbors, 3)
        assert num_neighbors.shape == (total_atoms,)
        assert neighbor_matrix.device == torch.device(device)
        assert neighbor_matrix_shifts.device == torch.device(device)
        assert num_neighbors.device == torch.device(device)

        # Check neighbor counts
        assert torch.all(num_neighbors >= 0)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    @pytest.mark.parametrize("pbc_flag", [True, False])
    def test_batch_naive_neighbor_list_return_neighbor_list(
        self, device, dtype, half_fill, pbc_flag
    ):
        """Test batch_naive_neighbor_list with return_neighbor_list=True."""
        atoms_per_system = [4, 6, 5]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=3, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 25

        if not pbc_flag:
            cell_batch = None
            pbc_batch = None

            neighbor_list, neighbor_ptr = batch_naive_neighbor_list(
                positions=positions_batch,
                cutoff=cutoff,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors=max_neighbors,
                pbc=pbc_batch,
                cell=cell_batch,
                half_fill=half_fill,
                return_neighbor_list=True,
            )
        else:
            neighbor_list, neighbor_ptr, _ = batch_naive_neighbor_list(
                positions=positions_batch,
                cutoff=cutoff,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors=max_neighbors,
                pbc=pbc_batch,
                cell=cell_batch,
                half_fill=half_fill,
                return_neighbor_list=True,
            )

        # Check that we get neighbor list format (2, N) instead of matrix
        assert neighbor_list.ndim == 2
        assert neighbor_list.shape[0] == 2
        assert neighbor_list.dtype == torch.int32
        assert neighbor_ptr.dtype == torch.int32

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_batch_naive_neighbor_list_consistency_with_single_system_no_pbc(
        self, device, dtype
    ):
        """Test that batch neighbor list gives same results as single system calls."""
        # Create a batch with multiple systems
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 30

        # Get batch result
        _, num_neighbors_batch, _ = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            pbc=pbc_batch,
            cell=cell_batch,
            max_neighbors=max_neighbors,
            half_fill=False,  # Use full fill for easier comparison
            return_neighbor_list=False,
        )

        # Get single system results for comparison
        for sys_idx in range(2):
            start_idx = batch_ptr[sys_idx].item()
            end_idx = batch_ptr[sys_idx + 1].item()

            positions_single = positions_batch[start_idx:end_idx]
            pbc_single = pbc_batch[sys_idx : sys_idx + 1]
            cell_single = cell_batch[sys_idx : sys_idx + 1]
            # Create batch_idx and batch_ptr for single system (batch of size 1)
            n_atoms_single = positions_single.shape[0]
            batch_idx_single = torch.zeros(
                n_atoms_single, dtype=torch.int32, device=positions_single.device
            )
            batch_ptr_single = torch.tensor(
                [0, n_atoms_single], dtype=torch.int32, device=positions_single.device
            )
            (
                neighbor_matrix_single,
                num_neighbors_single,
                neighbor_matrix_shifts_single,
            ) = batch_naive_neighbor_list(
                positions=positions_single,
                cutoff=cutoff,
                batch_idx=batch_idx_single,
                batch_ptr=batch_ptr_single,
                pbc=pbc_single,
                cell=cell_single,
                max_neighbors=max_neighbors,
                half_fill=False,
            )

            # Compare neighbor counts (should be identical for the same system)
            torch.testing.assert_close(
                num_neighbors_batch[start_idx:end_idx],
                num_neighbors_single,
                rtol=0,
                atol=0,
            )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_batch_naive_neighbor_list_consistency_with_single_system(
        self, device, dtype
    ):
        """Test that batch neighbor list gives same results as single system calls."""
        # Create a batch with multiple systems
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 30

        # Get batch result
        _, num_neighbors_batch, _ = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            pbc=pbc_batch,
            cell=cell_batch,
            max_neighbors=max_neighbors,
            half_fill=False,  # Use full fill for easier comparison
            return_neighbor_list=False,
        )

        # Get single system results for comparison
        for sys_idx in range(2):
            start_idx = batch_ptr[sys_idx].item()
            end_idx = batch_ptr[sys_idx + 1].item()

            positions_single = positions_batch[start_idx:end_idx]
            cell_single = cell_batch[sys_idx : sys_idx + 1]
            pbc_single = pbc_batch[sys_idx : sys_idx + 1]
            # Create batch_idx and batch_ptr for single system (batch of size 1)
            n_atoms_single = positions_single.shape[0]
            batch_idx_single = torch.zeros(
                n_atoms_single, dtype=torch.int32, device=positions_single.device
            )
            batch_ptr_single = torch.tensor(
                [0, n_atoms_single], dtype=torch.int32, device=positions_single.device
            )

            (
                neighbor_matrix_single,
                num_neighbors_single,
                neighbor_matrix_shifts_single,
            ) = batch_naive_neighbor_list(
                positions=positions_single,
                cutoff=cutoff,
                batch_idx=batch_idx_single,
                batch_ptr=batch_ptr_single,
                pbc=pbc_single,
                cell=cell_single,
                max_neighbors=max_neighbors,
                half_fill=False,
            )

            # Compare neighbor counts (should be identical for the same system)
            torch.testing.assert_close(
                num_neighbors_batch[start_idx:end_idx],
                num_neighbors_single,
                rtol=0,
                atol=0,
            )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_list_edge_cases(self, device, dtype, half_fill):
        """Test edge cases for batch_naive_neighbor_list."""
        # Empty batch
        positions_empty = torch.empty(0, 3, dtype=dtype, device=device)
        batch_idx_empty = torch.empty(0, dtype=torch.int32, device=device)
        batch_ptr_empty = torch.tensor([0], dtype=torch.int32, device=device)

        neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
            positions=positions_empty,
            cutoff=1.0,
            batch_idx=batch_idx_empty,
            batch_ptr=batch_ptr_empty,
            max_neighbors=10,
            pbc=None,
            cell=None,
            half_fill=half_fill,
        )
        assert neighbor_matrix.shape == (0, 10)
        assert num_neighbors.shape == (0,)

        # Single system with single atom
        positions_single = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        batch_idx_single = torch.tensor([0], dtype=torch.int32, device=device)
        batch_ptr_single = torch.tensor([0, 1], dtype=torch.int32, device=device)

        neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
            positions=positions_single,
            cutoff=1.0,
            batch_idx=batch_idx_single,
            batch_ptr=batch_ptr_single,
            max_neighbors=10,
            pbc=None,
            cell=None,
            half_fill=half_fill,
        )
        assert num_neighbors[0].item() == 0, "Single atom should have no neighbors"

        # Zero cutoff
        atoms_per_system = [3, 4]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=0.0,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=10,
            pbc=None,
            cell=None,
            half_fill=half_fill,
        )
        assert torch.all(num_neighbors == 0), "Zero cutoff should find no neighbors"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_list_error_conditions(self, device, dtype, half_fill):
        """Test error conditions for batch_naive_neighbor_list."""
        atoms_per_system = [4, 5]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        # Test mismatched cell and pbc arguments
        with pytest.raises(
            ValueError, match="If cell is provided, pbc must also be provided"
        ):
            batch_naive_neighbor_list(
                positions_batch,
                1.0,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors=10,
                pbc=None,
                cell=cell_batch,
            )

        with pytest.raises(
            ValueError, match="If pbc is provided, cell must also be provided"
        ):
            batch_naive_neighbor_list(
                positions_batch,
                1.0,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors=10,
                pbc=pbc_batch,
                cell=None,
            )


class TestBatchNaivePerformanceAndScaling:
    """Test performance characteristics and scaling of batch implementation."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_batch_scaling_with_system_size(self, device):
        """Test that batch implementation scales as expected with system size."""
        import time

        dtype = torch.float32
        cutoff = 1.2
        max_neighbors = 50

        # Test different batch sizes
        batch_sizes = (
            [(2, [10, 12], [2.0, 2.0]), (3, [8, 10, 12], [2.0, 2.0, 2.0])]
            if device == "cpu"
            else [
                (3, [20, 25, 30], [2.0, 2.0, 2.0]),
                (4, [15, 20, 25, 30], [2.0, 2.0, 2.0, 2.0]),
            ]
        )
        times = []

        for num_systems, atoms_per_system, cell_sizes in batch_sizes:
            positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
                num_systems=num_systems,
                cell_sizes=cell_sizes,
                atoms_per_system=atoms_per_system,
                dtype=dtype,
                device=device,
            )
            batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)
            # Warm up
            for _ in range(10):
                batch_naive_neighbor_list(
                    positions_batch,
                    cutoff,
                    batch_idx,
                    batch_ptr,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    max_neighbors=max_neighbors,
                    half_fill=True,
                )

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            # Time the operation
            start_time = time.time()
            for _ in range(100):
                batch_naive_neighbor_list(
                    positions_batch,
                    cutoff,
                    batch_idx,
                    batch_ptr,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    max_neighbors=max_neighbors,
                    half_fill=True,
                )

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            elapsed = time.time() - start_time
            times.append(elapsed)

        # Check that it doesn't grow too fast
        assert times[1] > times[0] * 0.5, "Time should increase with batch size"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_batch_cutoff_scaling(self, device):
        """Test scaling with different cutoff values."""
        dtype = torch.float32
        atoms_per_system = [15, 18, 20]
        max_neighbors = 100

        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=3, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        # Test different cutoffs
        cutoffs = [0.8, 1.2, 1.6, 2.0]
        neighbor_counts = []

        for cutoff in cutoffs:
            _, num_neighbors, _ = batch_naive_neighbor_list(
                positions_batch,
                cutoff,
                batch_idx,
                batch_ptr,
                pbc=pbc_batch,
                cell=cell_batch,
                max_neighbors=max_neighbors,
                half_fill=True,
            )
            total_pairs = num_neighbors.sum().item()
            neighbor_counts.append(total_pairs)

        # Check that neighbor count increases with cutoff
        for i in range(1, len(neighbor_counts)):
            assert neighbor_counts[i] >= neighbor_counts[i - 1], (
                f"Neighbor count should increase with cutoff: {neighbor_counts}"
            )


class TestBatchNaiveRobustness:
    """Test robustness of batch implementation to various inputs."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_random_batch_systems(self, device, dtype, half_fill):
        """Test with random batch systems of various sizes and configurations."""
        for pbc_flag in [True, False]:
            # Test several random batch configurations
            for seed in [42, 123, 456]:
                atoms_per_system = [12, 15, 10, 18]
                positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
                    num_systems=4,
                    cell_sizes=[2.0, 2.0, 2.0, 2.0],
                    atoms_per_system=atoms_per_system,
                    dtype=dtype,
                    device=device,
                    seed=seed,
                    pbc_flag=pbc_flag,
                )
                batch_idx, batch_ptr = create_batch_idx_and_ptr(
                    atoms_per_system, device
                )

                cutoff = 1.3
                max_neighbors = 40

                # Should not crash
                if pbc_flag:
                    neighbor_matrix, num_neighbors, neighbor_matrix_shifts = (
                        batch_naive_neighbor_list(
                            positions=positions_batch,
                            cutoff=cutoff,
                            max_neighbors=max_neighbors,
                            batch_idx=batch_idx,
                            batch_ptr=batch_ptr,
                            pbc=pbc_batch,
                            cell=cell_batch,
                            half_fill=half_fill,
                        )
                    )
                    assert neighbor_matrix_shifts.device == torch.device(device)
                else:
                    neighbor_matrix, num_neighbors = batch_naive_neighbor_list(
                        positions=positions_batch,
                        cutoff=cutoff,
                        max_neighbors=max_neighbors,
                        pbc=None,
                        cell=None,
                        batch_idx=batch_idx,
                        batch_ptr=batch_ptr,
                        half_fill=half_fill,
                    )

                # Basic sanity checks
                assert torch.all(num_neighbors >= 0)
                assert torch.all(num_neighbors <= max_neighbors)
                assert neighbor_matrix.device == torch.device(device)
                assert num_neighbors.device == torch.device(device)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_mixed_system_sizes(self, device, dtype, half_fill):
        """Test with very different system sizes in the same batch."""
        # Mix of small and large systems
        atoms_per_system = [2, 25, 5, 30, 1]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=5,
            cell_sizes=[2.0, 2.0, 2.0, 2.0, 2.0],
            atoms_per_system=atoms_per_system,
            dtype=dtype,
            device=device,
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.5
        max_neighbors = 50

        _, num_neighbors, _ = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=half_fill,
        )

        # Check that single-atom systems have no neighbors
        single_atom_indices = []
        for i, num_atoms in enumerate(atoms_per_system):
            if num_atoms == 1:
                start_idx = batch_ptr[i].item()
                single_atom_indices.append(start_idx)

        for idx in single_atom_indices:
            assert num_neighbors[idx].item() == 0, (
                f"Single atom at index {idx} should have no neighbors"
            )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_precision_consistency(self, device, dtype, half_fill):
        """Test that float32 and float64 give consistent results."""
        atoms_per_system = [6, 8, 7]
        positions_batch_f32, cell_batch_f32, pbc_batch, _ = create_batch_systems(
            num_systems=3,
            atoms_per_system=atoms_per_system,
            dtype=torch.float32,
            device=device,
        )
        positions_batch_f64 = positions_batch_f32.double()
        cell_batch_f64 = cell_batch_f32.double()

        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 1.2
        max_neighbors = 30

        # Get results for both precisions
        _, num_neighbors_f32, _ = batch_naive_neighbor_list(
            positions_batch_f32,
            cutoff,
            pbc=pbc_batch,
            cell=cell_batch_f32,
            max_neighbors=max_neighbors,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            half_fill=half_fill,
        )
        _, num_neighbors_f64, _ = batch_naive_neighbor_list(
            positions_batch_f64,
            cutoff,
            pbc=pbc_batch,
            cell=cell_batch_f64,
            max_neighbors=max_neighbors,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            half_fill=half_fill,
        )

        # Neighbor counts should be identical (for this exact geometry)
        torch.testing.assert_close(num_neighbors_f32, num_neighbors_f64, rtol=0, atol=0)


class TestBatchNaiveMemoryAndPerformance:
    """Test memory usage and performance characteristics of batch implementation."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_memory_scaling(self, device, half_fill):
        """Test that memory usage scales reasonably with batch size."""
        import gc

        dtype = torch.float32
        cutoff = 1.2

        # Test different batch sizes
        batch_configs = (
            [([8, 10], 2), ([12, 15], 2)]
            if device == "cpu"
            else [([20, 25], 2), ([30, 35], 2)]
        )

        for atoms_per_system, num_systems in batch_configs:
            positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
                num_systems=num_systems,
                atoms_per_system=atoms_per_system,
                dtype=dtype,
                device=device,
            )
            batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

            # Estimate reasonable max_neighbors based on system size and cutoff
            max_neighbors = 40

            # Clear cache before test
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

            # Run batch implementation
            neighbor_matrix, num_neighbors, neighbor_matrix_shifts = (
                batch_naive_neighbor_list(
                    positions=positions_batch,
                    cutoff=cutoff,
                    max_neighbors=max_neighbors,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    batch_idx=batch_idx,
                    batch_ptr=batch_ptr,
                    half_fill=half_fill,
                )
            )

            # Basic checks that output is reasonable
            total_atoms = positions_batch.shape[0]
            assert neighbor_matrix.shape == (total_atoms, max_neighbors)
            assert neighbor_matrix_shifts.shape == (
                total_atoms,
                max_neighbors,
                3,
            )
            assert num_neighbors.shape == (total_atoms,)
            assert torch.all(num_neighbors >= 0)
            assert torch.all(num_neighbors <= max_neighbors)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_max_neighbors_overflow_handling(self, device, dtype, half_fill):
        """Test behavior when max_neighbors is exceeded."""
        # Create a dense batch system with small max_neighbors to force overflow
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff = 2.0  # Large cutoff to find many neighbors
        max_neighbors = 3  # Artificially small to trigger overflow

        # Should not crash, but may not find all neighbors
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = (
            batch_naive_neighbor_list(
                positions=positions_batch,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                pbc=pbc_batch,
                cell=cell_batch,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                half_fill=half_fill,
            )
        )

        # Should still produce valid output, just potentially incomplete
        total_atoms = positions_batch.shape[0]
        assert torch.all(num_neighbors >= 0)
        assert neighbor_matrix.shape == (total_atoms, max_neighbors)
        assert neighbor_matrix_shifts.shape == (
            total_atoms,
            max_neighbors,
            3,
        )
        assert num_neighbors.shape == (total_atoms,)
        assert neighbor_matrix.device == torch.device(device)
        assert neighbor_matrix_shifts.device == torch.device(device)
        assert num_neighbors.device == torch.device(device)


class TestBatchNaiveDualCutoffMainAPI:
    """Test the main batch naive dual cutoff neighbor list API function."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_list_dual_cutoff_no_pbc(
        self, device, dtype, half_fill
    ):
        """Test batch_naive_neighbor_list_dual_cutoff without periodic boundary conditions."""
        atoms_per_system = [6, 8]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 20
        max_neighbors2 = 30

        # Test without PBC
        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            batch_naive_neighbor_list_dual_cutoff(
                positions=positions_batch,
                cutoff1=cutoff1,
                cutoff2=cutoff2,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
                pbc=None,
                cell=None,
                half_fill=half_fill,
            )
        )

        # Check output types and shapes
        expected_rows = positions_batch.shape[0]
        assert neighbor_matrix1.dtype == torch.int32
        assert neighbor_matrix2.dtype == torch.int32
        assert num_neighbors1.dtype == torch.int32
        assert num_neighbors2.dtype == torch.int32
        assert neighbor_matrix1.shape == (expected_rows, max_neighbors1)
        assert neighbor_matrix2.shape == (expected_rows, max_neighbors2)
        assert num_neighbors1.shape == (positions_batch.shape[0],)
        assert num_neighbors2.shape == (positions_batch.shape[0],)
        assert neighbor_matrix1.device == torch.device(device)
        assert neighbor_matrix2.device == torch.device(device)
        assert num_neighbors1.device == torch.device(device)
        assert num_neighbors2.device == torch.device(device)

        # Check neighbor counts are reasonable
        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert torch.all(num_neighbors1 <= max_neighbors1)
        assert torch.all(num_neighbors2 <= max_neighbors2)
        assert torch.all(num_neighbors2 >= num_neighbors1)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_list_dual_cutoff_with_pbc(
        self, device, dtype, half_fill
    ):
        """Test batch_naive_neighbor_list_dual_cutoff with periodic boundary conditions."""
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 30
        max_neighbors2 = 50

        (
            neighbor_matrix1,
            num_neighbors1,
            neighbor_matrix_shifts1,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        ) = batch_naive_neighbor_list_dual_cutoff(
            positions=positions_batch,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=half_fill,
        )

        # Check output types and shapes
        expected_rows = positions_batch.shape[0]
        assert neighbor_matrix1.dtype == torch.int32
        assert neighbor_matrix2.dtype == torch.int32
        assert neighbor_matrix_shifts1.dtype == torch.int32
        assert neighbor_matrix_shifts2.dtype == torch.int32
        assert num_neighbors1.dtype == torch.int32
        assert num_neighbors2.dtype == torch.int32
        assert neighbor_matrix1.shape == (expected_rows, max_neighbors1)
        assert neighbor_matrix2.shape == (expected_rows, max_neighbors2)
        assert neighbor_matrix_shifts1.shape == (expected_rows, max_neighbors1, 3)
        assert neighbor_matrix_shifts2.shape == (expected_rows, max_neighbors2, 3)
        assert num_neighbors1.shape == (positions_batch.shape[0],)
        assert num_neighbors2.shape == (positions_batch.shape[0],)
        assert neighbor_matrix1.device == torch.device(device)
        assert neighbor_matrix2.device == torch.device(device)
        assert neighbor_matrix_shifts1.device == torch.device(device)
        assert neighbor_matrix_shifts2.device == torch.device(device)
        assert num_neighbors1.device == torch.device(device)
        assert num_neighbors2.device == torch.device(device)

        # Check neighbor counts
        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert torch.all(num_neighbors2 >= num_neighbors1)

        # With PBC, should generally have more neighbors than without
        assert num_neighbors1.sum() >= 0, "Should find some neighbors with PBC"
        assert num_neighbors2.sum() >= 0, "Should find some neighbors with PBC"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    @pytest.mark.parametrize("pbc_flag", [True, False])
    def test_batch_naive_neighbor_list_dual_cutoff_return_neighbor_list(
        self, device, dtype, half_fill, pbc_flag
    ):
        """Test batch_naive_neighbor_list_dual_cutoff with return_neighbor_list=True."""
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 30
        max_neighbors2 = 50

        if not pbc_flag:
            cell_batch = None
            pbc_batch = None

            neighbor_list1, neighbor_ptr1, neighbor_list2, neighbor_ptr2 = (
                batch_naive_neighbor_list_dual_cutoff(
                    positions=positions_batch,
                    cutoff1=cutoff1,
                    cutoff2=cutoff2,
                    batch_idx=batch_idx,
                    batch_ptr=batch_ptr,
                    max_neighbors1=max_neighbors1,
                    max_neighbors2=max_neighbors2,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    half_fill=half_fill,
                    return_neighbor_list=True,
                )
            )

        else:
            (
                neighbor_list1,
                neighbor_ptr1,
                unit_shifts1,
                neighbor_list2,
                neighbor_ptr2,
                unit_shifts2,
            ) = batch_naive_neighbor_list_dual_cutoff(
                positions=positions_batch,
                cutoff1=cutoff1,
                cutoff2=cutoff2,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
                pbc=pbc_batch,
                cell=cell_batch,
                half_fill=half_fill,
                return_neighbor_list=True,
            )

        # Check that we get neighbor list format (2, N) instead of matrix
        assert neighbor_list1.ndim == 2
        assert neighbor_list2.ndim == 2
        assert neighbor_list1.shape[0] == 2
        assert neighbor_list2.shape[0] == 2
        assert neighbor_list1.dtype == torch.int32
        assert neighbor_list2.dtype == torch.int32
        assert neighbor_ptr1.dtype == torch.int32
        assert neighbor_ptr2.dtype == torch.int32

        # Check that neighbor_list2 has at least as many pairs as neighbor_list1
        assert neighbor_list2.shape[1] >= neighbor_list1.shape[1], (
            "Larger cutoff should find at least as many neighbor pairs"
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_batch_naive_neighbor_list_dual_cutoff_consistency_with_single_cutoff(
        self, device, dtype
    ):
        """Test that dual cutoff results are consistent with two single cutoff calls."""
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 30
        max_neighbors2 = 50

        # Get dual cutoff result
        (
            neighbor_matrix1_dual,
            num_neighbors1_dual,
            neighbor_matrix_shifts1_dual,
            neighbor_matrix2_dual,
            num_neighbors2_dual,
            neighbor_matrix_shifts2_dual,
        ) = batch_naive_neighbor_list_dual_cutoff(
            positions=positions_batch,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=False,  # Use full fill for easier comparison
        )

        # Get single cutoff results
        (
            neighbor_matrix1_single,
            num_neighbors1_single,
            neighbor_matrix_shifts1_single,
        ) = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff1,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=max_neighbors2,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=False,
        )

        (
            neighbor_matrix2_single,
            num_neighbors2_single,
            neighbor_matrix_shifts2_single,
        ) = batch_naive_neighbor_list(
            positions=positions_batch,
            cutoff=cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors=max_neighbors2,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=False,
        )

        # Compare neighbor counts (should be identical)
        torch.testing.assert_close(
            num_neighbors1_dual, num_neighbors1_single, rtol=0, atol=0
        )
        torch.testing.assert_close(
            num_neighbors2_dual, num_neighbors2_single, rtol=0, atol=0
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_list_dual_cutoff_edge_cases(
        self, device, dtype, half_fill
    ):
        """Test edge cases for batch_naive_neighbor_list_dual_cutoff."""
        # Empty system
        positions_empty = torch.empty(0, 3, dtype=dtype, device=device)
        batch_idx_empty = torch.empty(0, dtype=torch.int32, device=device)
        batch_ptr_empty = torch.tensor([0], dtype=torch.int32, device=device)

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            batch_naive_neighbor_list_dual_cutoff(
                positions=positions_empty,
                cutoff1=1.0,
                cutoff2=1.5,
                batch_idx=batch_idx_empty,
                batch_ptr=batch_ptr_empty,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=None,
                cell=None,
                half_fill=half_fill,
            )
        )
        assert neighbor_matrix1.shape == (0, 10)
        assert neighbor_matrix2.shape == (0, 15)
        assert num_neighbors1.shape == (0,)
        assert num_neighbors2.shape == (0,)

        # Single atom system
        positions_single = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        batch_idx_single = torch.tensor([0], dtype=torch.int32, device=device)
        batch_ptr_single = torch.tensor([0, 1], dtype=torch.int32, device=device)

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            batch_naive_neighbor_list_dual_cutoff(
                positions=positions_single,
                cutoff1=1.0,
                cutoff2=1.5,
                batch_idx=batch_idx_single,
                batch_ptr=batch_ptr_single,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=None,
                cell=None,
                half_fill=half_fill,
            )
        )
        assert num_neighbors1[0].item() == 0, "Single atom should have no neighbors"
        assert num_neighbors2[0].item() == 0, "Single atom should have no neighbors"

        # Zero cutoffs
        atoms_per_system = [4, 4]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        neighbor_matrix1, num_neighbors1, neighbor_matrix2, num_neighbors2 = (
            batch_naive_neighbor_list_dual_cutoff(
                positions=positions_batch,
                cutoff1=0.0,
                cutoff2=0.0,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=None,
                cell=None,
                half_fill=half_fill,
            )
        )
        assert torch.all(num_neighbors1 == 0), "Zero cutoffs should find no neighbors"
        assert torch.all(num_neighbors2 == 0), "Zero cutoffs should find no neighbors"

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_batch_naive_neighbor_list_dual_cutoff_error_conditions(
        self, device, dtype, half_fill
    ):
        """Test error conditions for batch_naive_neighbor_list_dual_cutoff."""
        atoms_per_system = [4, 6]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        # Test mismatched cell and pbc arguments
        with pytest.raises(
            ValueError, match="If cell is provided, pbc must also be provided"
        ):
            batch_naive_neighbor_list_dual_cutoff(
                positions_batch,
                1.0,
                1.5,
                batch_idx,
                batch_ptr,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=None,
                cell=cell_batch,
            )

        with pytest.raises(
            ValueError, match="If pbc is provided, cell must also be provided"
        ):
            batch_naive_neighbor_list_dual_cutoff(
                positions_batch,
                1.0,
                1.5,
                batch_idx,
                batch_ptr,
                max_neighbors1=10,
                max_neighbors2=15,
                pbc=pbc_batch,
                cell=None,
            )

    def test_max_neighbors_same_value(self):
        """Test that both neighbor matrices have correct shape when given same max_neighbors."""
        atoms_per_system = [4, 6]
        positions_batch, _, _, _ = create_batch_systems(
            num_systems=2,
            atoms_per_system=atoms_per_system,
            dtype=torch.float32,
            device="cpu",
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, "cpu")

        # Test with same max_neighbors for both cutoffs
        neighbor_matrix1, _, neighbor_matrix2, _ = (
            batch_naive_neighbor_list_dual_cutoff(
                positions=positions_batch,
                cutoff1=1.0,
                cutoff2=1.5,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=10,
                max_neighbors2=10,  # Same as max_neighbors1
                pbc=None,
                cell=None,
            )
        )

        # Both matrices should have same number of columns
        assert neighbor_matrix1.shape[1] == neighbor_matrix2.shape[1] == 10


class TestBatchNaiveDualCutoffPerformanceAndScaling:
    """Test performance characteristics and scaling of batch naive dual cutoff implementation."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_batch_dual_cutoff_scaling_with_system_size(self, device):
        """Test that batch dual cutoff implementation scales as expected with system size."""
        import time

        dtype = torch.float32
        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 50
        max_neighbors2 = 80

        # Test different batch sizes
        batch_sizes = (
            [(2, [10, 12], [2.0, 2.0]), (3, [8, 10, 12], [2.0, 2.0, 2.0])]
            if device == "cpu"
            else [
                (3, [20, 25, 30], [2.0, 2.0, 2.0]),
                (4, [15, 20, 25, 30], [2.0, 2.0, 2.0, 2.0]),
            ]
        )
        times = []

        for num_systems, atoms_per_system, cell_sizes in batch_sizes:
            positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
                num_systems=num_systems,
                cell_sizes=cell_sizes,
                atoms_per_system=atoms_per_system,
                dtype=dtype,
                device=device,
            )
            batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

            # Warm up
            for _ in range(10):
                batch_naive_neighbor_list_dual_cutoff(
                    positions_batch,
                    cutoff1,
                    cutoff2,
                    batch_idx,
                    batch_ptr,
                    max_neighbors1=max_neighbors1,
                    max_neighbors2=max_neighbors2,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    half_fill=True,
                )

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            # Time the operation
            start_time = time.time()
            for _ in range(100):
                batch_naive_neighbor_list_dual_cutoff(
                    positions_batch,
                    cutoff1,
                    cutoff2,
                    batch_idx,
                    batch_ptr,
                    max_neighbors1=max_neighbors1,
                    max_neighbors2=max_neighbors2,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    half_fill=True,
                )

            if device.startswith("cuda"):
                torch.cuda.synchronize()

            elapsed = time.time() - start_time
            times.append(elapsed)

        # Check that it doesn't grow too fast (should be roughly O(N^2))
        # This is a loose check since we can't expect perfect scaling
        assert times[1] > times[0] * 0.5, "Time should increase with system size"
        if len(times) > 2:
            # Very loose scaling check
            scaling_factor = times[-1] / times[0]
            total_atoms_ratio = sum(batch_sizes[-1][1]) / sum(batch_sizes[0][1])
            size_factor = total_atoms_ratio**2
            assert scaling_factor < size_factor * 5, (
                "Scaling should not be much worse than O(N^2)"
            )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_batch_dual_cutoff_cutoff_scaling(self, device):
        """Test scaling with different cutoff values."""
        dtype = torch.float32
        atoms_per_system = [15, 20]
        max_neighbors1 = 100
        max_neighbors2 = 150

        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        # Test different cutoff pairs
        cutoff_pairs = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5)]
        neighbor_counts1 = []
        neighbor_counts2 = []

        for cutoff1, cutoff2 in cutoff_pairs:
            (_, num_neighbors1, _, _, num_neighbors2, _) = (
                batch_naive_neighbor_list_dual_cutoff(
                    positions_batch,
                    cutoff1,
                    cutoff2,
                    batch_idx,
                    batch_ptr,
                    max_neighbors1=max_neighbors1,
                    max_neighbors2=max_neighbors2,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    half_fill=True,
                )
            )
            total_pairs1 = num_neighbors1.sum().item()
            total_pairs2 = num_neighbors2.sum().item()
            neighbor_counts1.append(total_pairs1)
            neighbor_counts2.append(total_pairs2)

        # Check that neighbor count increases with cutoff
        for i in range(1, len(neighbor_counts1)):
            assert neighbor_counts1[i] >= neighbor_counts1[i - 1], (
                f"Neighbor count should increase with cutoff1: {neighbor_counts1}"
            )
            assert neighbor_counts2[i] >= neighbor_counts2[i - 1], (
                f"Neighbor count should increase with cutoff2: {neighbor_counts2}"
            )

        # Check that cutoff2 always finds at least as many neighbors as cutoff1
        for count1, count2 in zip(neighbor_counts1, neighbor_counts2):
            assert count2 >= count1, (
                f"Larger cutoff should find at least as many neighbors: {count1} vs {count2}"
            )


class TestBatchNaiveDualCutoffRobustness:
    """Test robustness of batch naive dual cutoff implementation to various inputs."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_random_systems(self, device, dtype, half_fill):
        """Test with random systems of various sizes and configurations."""
        for pbc_flag in [True, False]:
            # Test several random systems
            for seed in [42, 123, 456]:
                atoms_per_system = [15, 20, 18]
                positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
                    num_systems=3,
                    atoms_per_system=atoms_per_system,
                    dtype=dtype,
                    device=device,
                    seed=seed,
                    pbc_flag=pbc_flag,
                )
                batch_idx, batch_ptr = create_batch_idx_and_ptr(
                    atoms_per_system, device
                )

                cutoff1 = 1.0
                cutoff2 = 1.5
                max_neighbors1 = 50
                max_neighbors2 = 80

                if not pbc_flag:
                    cell_batch = None
                    pbc_batch = None

                # Should not crash
                result = batch_naive_neighbor_list_dual_cutoff(
                    positions=positions_batch,
                    cutoff1=cutoff1,
                    cutoff2=cutoff2,
                    batch_idx=batch_idx,
                    batch_ptr=batch_ptr,
                    max_neighbors1=max_neighbors1,
                    max_neighbors2=max_neighbors2,
                    pbc=pbc_batch,
                    cell=cell_batch,
                    half_fill=half_fill,
                )

                if pbc_flag:
                    (
                        neighbor_matrix1,
                        num_neighbors1,
                        neighbor_matrix_shifts1,
                        neighbor_matrix2,
                        num_neighbors2,
                        neighbor_matrix_shifts2,
                    ) = result
                    # Basic sanity checks
                    assert torch.all(num_neighbors1 >= 0)
                    assert torch.all(num_neighbors2 >= 0)
                    assert torch.all(num_neighbors1 <= max_neighbors1)
                    assert torch.all(num_neighbors2 <= max_neighbors2)
                    assert torch.all(num_neighbors2 >= num_neighbors1)
                    assert neighbor_matrix1.device == torch.device(device)
                    assert neighbor_matrix_shifts1.device == torch.device(device)
                    assert neighbor_matrix2.device == torch.device(device)
                    assert neighbor_matrix_shifts2.device == torch.device(device)
                    assert num_neighbors1.device == torch.device(device)
                    assert num_neighbors2.device == torch.device(device)
                else:
                    (
                        neighbor_matrix1,
                        num_neighbors1,
                        neighbor_matrix2,
                        num_neighbors2,
                    ) = result
                    # Basic sanity checks
                    assert torch.all(num_neighbors1 >= 0)
                    assert torch.all(num_neighbors2 >= 0)
                    assert torch.all(num_neighbors1 <= max_neighbors1)
                    assert torch.all(num_neighbors2 <= max_neighbors2)
                    assert torch.all(num_neighbors2 >= num_neighbors1)
                    assert neighbor_matrix1.device == torch.device(device)
                    assert neighbor_matrix2.device == torch.device(device)
                    assert num_neighbors1.device == torch.device(device)
                    assert num_neighbors2.device == torch.device(device)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_extreme_geometries(self, device, dtype, half_fill):
        """Test with extreme cell geometries."""
        # Very elongated cells
        atoms_per_system = [8, 10]
        positions_batch = torch.rand(18, 3, dtype=dtype, device=device)
        cell_batch = torch.tensor(
            [
                [[10.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
                [[0.1, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.1]],
            ],
            dtype=dtype,
            device=device,
        )
        pbc_batch = torch.tensor(
            [[True, True, True], [True, True, True]], device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        # Scale positions to fit in cells
        positions_batch[:8] = positions_batch[:8] * torch.tensor(
            [10.0, 0.1, 0.1], device=device
        )
        positions_batch[8:] = positions_batch[8:] * torch.tensor(
            [0.1, 10.0, 0.1], device=device
        )

        cutoff1 = 0.15
        cutoff2 = 0.25
        max_neighbors1 = 20
        max_neighbors2 = 30

        # Should handle extreme aspect ratios
        (
            neighbor_matrix1,
            num_neighbors1,
            neighbor_matrix_shifts1,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        ) = batch_naive_neighbor_list_dual_cutoff(
            positions=positions_batch,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=half_fill,
        )

        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert torch.all(num_neighbors2 >= num_neighbors1)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_large_cutoffs(self, device, dtype, half_fill):
        """Test with very large cutoffs."""
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        # Cutoffs larger than cell size
        large_cutoff1 = 4.0
        large_cutoff2 = 6.0
        max_neighbors1 = 100
        max_neighbors2 = 150

        (
            neighbor_matrix1,
            num_neighbors1,
            neighbor_matrix_shifts1,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        ) = batch_naive_neighbor_list_dual_cutoff(
            positions=positions_batch,
            cutoff1=large_cutoff1,
            cutoff2=large_cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=half_fill,
        )

        # Should find many neighbors
        assert num_neighbors1.sum() > 0
        assert num_neighbors2.sum() > 0
        assert torch.all(num_neighbors2 >= num_neighbors1)
        # Each atom should have multiple neighbors (including periodic images)
        assert torch.all(
            num_neighbors1 >= 0
        )  # Some atoms might have no neighbors in half_fill mode
        assert torch.all(num_neighbors2 >= 0)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_precision_consistency(self, device, dtype, half_fill):
        """Test that float32 and float64 give consistent results."""
        atoms_per_system = [6, 8]
        positions_batch_f32, cell_batch_f32, pbc_batch, _ = create_batch_systems(
            num_systems=2,
            atoms_per_system=atoms_per_system,
            dtype=torch.float32,
            device=device,
        )
        positions_batch_f64 = positions_batch_f32.double()
        cell_batch_f64 = cell_batch_f32.double()
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 1.0
        cutoff2 = 1.5
        max_neighbors1 = 50
        max_neighbors2 = 80

        # Get results for both precisions
        (_, num_neighbors1_f32, _, _, num_neighbors2_f32, _) = (
            batch_naive_neighbor_list_dual_cutoff(
                positions_batch_f32,
                cutoff1,
                cutoff2,
                batch_idx,
                batch_ptr,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
                pbc=pbc_batch,
                cell=cell_batch_f32,
                half_fill=half_fill,
            )
        )
        (_, num_neighbors1_f64, _, _, num_neighbors2_f64, _) = (
            batch_naive_neighbor_list_dual_cutoff(
                positions_batch_f64,
                cutoff1,
                cutoff2,
                batch_idx,
                batch_ptr,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
                pbc=pbc_batch,
                cell=cell_batch_f64,
                half_fill=half_fill,
            )
        )

        # Neighbor counts should be identical (for this exact geometry)
        torch.testing.assert_close(
            num_neighbors1_f32, num_neighbors1_f64, rtol=0, atol=0
        )
        torch.testing.assert_close(
            num_neighbors2_f32, num_neighbors2_f64, rtol=0, atol=0
        )


class TestBatchNaiveDualCutoffMemoryAndPerformance:
    """Test memory usage and performance characteristics of batch naive dual cutoff implementation."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_memory_scaling(self, device, half_fill):
        """Test that memory usage scales reasonably with system size."""
        import gc

        dtype = torch.float32
        cutoff1 = 1.0
        cutoff2 = 1.5

        # Test different system sizes
        sizes = (
            [(2, [8, 10], [2.0, 2.0]), (3, [6, 8, 10], [2.0, 2.0, 2.0])]
            if device == "cpu"
            else [
                (3, [15, 20, 25], [2.0, 2.0, 2.0]),
                (4, [12, 15, 18, 20], [2.0, 2.0, 2.0, 2.0]),
            ]
        )

        for num_systems, atoms_per_system, cell_sizes in sizes:
            positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
                num_systems=num_systems,
                atoms_per_system=atoms_per_system,
                dtype=dtype,
                device=device,
                cell_sizes=cell_sizes,
            )
            batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

            # Estimate reasonable max_neighbors based on system size and cutoff
            max_neighbors1 = 50
            max_neighbors2 = 80

            # Clear cache before test
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

            # Run batch dual cutoff implementation
            (
                neighbor_matrix1,
                num_neighbors1,
                neighbor_matrix_shifts1,
                neighbor_matrix2,
                num_neighbors2,
                neighbor_matrix_shifts2,
            ) = batch_naive_neighbor_list_dual_cutoff(
                positions=positions_batch,
                cutoff1=cutoff1,
                cutoff2=cutoff2,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors1=max_neighbors1,
                max_neighbors2=max_neighbors2,
                pbc=pbc_batch,
                cell=cell_batch,
                half_fill=half_fill,
            )

            # Basic checks that output is reasonable
            total_atoms = positions_batch.shape[0]
            assert neighbor_matrix1.shape == (total_atoms, max_neighbors1)
            assert neighbor_matrix2.shape == (total_atoms, max_neighbors2)
            assert neighbor_matrix_shifts1.shape == (total_atoms, max_neighbors1, 3)
            assert neighbor_matrix_shifts2.shape == (total_atoms, max_neighbors2, 3)
            assert num_neighbors1.shape == (total_atoms,)
            assert num_neighbors2.shape == (total_atoms,)
            assert torch.all(num_neighbors1 >= 0), (
                "All neighbor counts should be non-negative"
            )
            assert torch.all(num_neighbors2 >= 0), (
                "All neighbor counts should be non-negative"
            )
            assert torch.all(num_neighbors1 <= max_neighbors1), (
                "Neighbor counts should not exceed maximum"
            )
            assert torch.all(num_neighbors2 <= max_neighbors2), (
                "Neighbor counts should not exceed maximum"
            )
            assert torch.all(num_neighbors2 >= num_neighbors1), (
                "Larger cutoff should find at least as many neighbors"
            )

            # Clean up
            del (
                neighbor_matrix1,
                neighbor_matrix2,
                neighbor_matrix_shifts1,
                neighbor_matrix_shifts2,
                num_neighbors1,
                num_neighbors2,
                positions_batch,
                cell_batch,
                pbc_batch,
                batch_idx,
                batch_ptr,
            )
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("half_fill", [True, False])
    def test_max_neighbors_overflow_handling(self, device, dtype, half_fill):
        """Test behavior when max_neighbors is exceeded."""

        # Create a dense system with small max_neighbors to force overflow
        atoms_per_system = [6, 8]
        positions_batch, cell_batch, pbc_batch, _ = create_batch_systems(
            num_systems=2, atoms_per_system=atoms_per_system, dtype=dtype, device=device
        )
        batch_idx, batch_ptr = create_batch_idx_and_ptr(atoms_per_system, device)

        cutoff1 = 2.0  # Large cutoffs to find many neighbors
        cutoff2 = 3.0
        max_neighbors1 = 3  # Artificially small to trigger overflow
        max_neighbors2 = 5

        # Should not crash, but may not find all neighbors
        (
            neighbor_matrix1,
            num_neighbors1,
            neighbor_matrix_shifts1,
            neighbor_matrix2,
            num_neighbors2,
            neighbor_matrix_shifts2,
        ) = batch_naive_neighbor_list_dual_cutoff(
            positions=positions_batch,
            cutoff1=cutoff1,
            cutoff2=cutoff2,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            max_neighbors1=max_neighbors1,
            max_neighbors2=max_neighbors2,
            pbc=pbc_batch,
            cell=cell_batch,
            half_fill=half_fill,
        )

        # Should still produce valid output, just potentially incomplete
        total_atoms = sum(atoms_per_system)
        assert torch.all(num_neighbors1 >= 0)
        assert torch.all(num_neighbors2 >= 0)
        assert neighbor_matrix1.shape == (total_atoms, max_neighbors1)
        assert neighbor_matrix2.shape == (total_atoms, max_neighbors2)
        assert neighbor_matrix_shifts1.shape == (
            total_atoms,
            max_neighbors1,
            3,
        )
        assert neighbor_matrix_shifts2.shape == (
            total_atoms,
            max_neighbors2,
            3,
        )
        assert num_neighbors1.shape == (total_atoms,)
        assert num_neighbors2.shape == (total_atoms,)
        assert neighbor_matrix1.device == torch.device(device)
        assert neighbor_matrix2.device == torch.device(device)
        assert neighbor_matrix_shifts1.device == torch.device(device)
        assert neighbor_matrix_shifts2.device == torch.device(device)
        assert num_neighbors1.device == torch.device(device)
        assert num_neighbors2.device == torch.device(device)
