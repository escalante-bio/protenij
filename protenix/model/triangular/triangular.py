# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Consolidated triangular attention and multiplicative update modules.

This module re-exports triangular operation components that were previously in
openfold_local/model/triangular_attention.py and
openfold_local/model/triangular_multiplicative_update.py.
"""

from protenix.openfold_local.model.triangular_attention import (
    TriangleAttention,
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from protenix.openfold_local.model.triangular_multiplicative_update import (
    BaseTriangleMultiplicativeUpdate,
    FusedTriangleMultiplicativeUpdate,
    FusedTriangleMultiplicationIncoming,
    FusedTriangleMultiplicationOutgoing,
    TriangleMultiplicativeUpdate,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)

__all__ = [
    "BaseTriangleMultiplicativeUpdate",
    "FusedTriangleMultiplicativeUpdate",
    "FusedTriangleMultiplicationIncoming",
    "FusedTriangleMultiplicationOutgoing",
    "TriangleAttention",
    "TriangleAttentionEndingNode",
    "TriangleAttentionStartingNode",
    "TriangleMultiplicativeUpdate",
    "TriangleMultiplicationIncoming",
    "TriangleMultiplicationOutgoing",
]
