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
Consolidated layer primitives for triangular operations.

This module re-exports layer components that were previously spread across
openfold_local/model/primitives.py, openfold_local/model/dropout.py, and
openfold_local/model/outer_product_mean.py.
"""

from protenix.openfold_local.model.primitives import (
    Attention,
    LayerNorm,
    Linear,
    OpenFoldLayerNorm,
    final_init_,
    gating_init_,
    glorot_uniform_init_,
    he_normal_init_,
    lecun_normal_init_,
    normal_init_,
    trunc_normal_init_,
)
from protenix.openfold_local.model.dropout import (
    Dropout,
    DropoutColumnwise,
    DropoutRowwise,
)
from protenix.openfold_local.model.outer_product_mean import OuterProductMean

__all__ = [
    "Attention",
    "Dropout",
    "DropoutColumnwise",
    "DropoutRowwise",
    "LayerNorm",
    "Linear",
    "OpenFoldLayerNorm",
    "OuterProductMean",
    "final_init_",
    "gating_init_",
    "glorot_uniform_init_",
    "he_normal_init_",
    "lecun_normal_init_",
    "normal_init_",
    "trunc_normal_init_",
]
