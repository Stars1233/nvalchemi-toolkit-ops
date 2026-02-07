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

"""Shared pytest fixtures for torch neighbors binding tests."""

from importlib import import_module

import pytest
import torch

# Check if vesin is available for consistency checks
try:
    _ = import_module("vesin")
    VESIN_AVAILABLE = True
except ModuleNotFoundError:
    VESIN_AVAILABLE = False

# Pytest marker for tests that require vesin
requires_vesin = pytest.mark.skipif(
    not VESIN_AVAILABLE, reason="`vesin` required for consistency checks."
)


@pytest.fixture(params=[torch.float32, torch.float64], ids=["float32", "float64"])
def dtype(request):
    """Fixture providing torch dtypes for testing.

    Returns
    -------
    torch.dtype
        The torch dtype (float32 or float64)
    """
    return request.param
