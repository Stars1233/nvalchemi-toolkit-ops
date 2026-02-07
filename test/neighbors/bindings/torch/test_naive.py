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

"""Tests for PyTorch bindings of naive neighbor list methods."""

import pytest
import torch

from nvalchemiops.torch.neighbors.naive import (
    naive_neighbor_list,
)
from nvalchemiops.torch.neighbors.neighbor_utils import (
    compute_naive_num_shifts,
)

from ...test_utils import (
    assert_neighbor_lists_equal,
    brute_force_neighbors,
    create_random_system,
    create_simple_cubic_system,
)
from .conftest import requires_vesin


class TestNaiveMainAPI:
    """Test the main naive neighbor list API function."""

    @requires_vesin
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
            # TODO: this test depends on runtime performance and
            # so it might be flaky
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
