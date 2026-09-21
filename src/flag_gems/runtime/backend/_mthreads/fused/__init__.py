# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .cross_entropy_loss import cross_entropy_loss
from .sparse_attention import sparse_attn_triton

# matmuladd / matmul_bias_activation 依赖 triton.tools.tensor_descriptor
# （triton 3.3+），musa triton 3.2 无此模块；缺依赖时只跳过这两个，
# 避免拖垮整个 fused 包的导入
try:
    from .matmul_bias_activation import matmul_bias_activation
    from .matmuladd import matmuladd

    _HAS_FUSED_MATMUL = True
except ModuleNotFoundError:
    _HAS_FUSED_MATMUL = False

__all__ = [
    "cross_entropy_loss",
    "sparse_attn_triton",
]

if _HAS_FUSED_MATMUL:
    __all__.extend(["matmul_bias_activation", "matmuladd"])
