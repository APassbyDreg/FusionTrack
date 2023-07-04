# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
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

# from .multi_scale_deform_attn import (
#     MultiScaleDeformableAttention,
#     multi_scale_deformable_attn_pytorch,
# )

# -------------------------------- debug only -------------------------------- #
try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention as MMCV_MSDA
except ImportError:
    import warnings
    warnings.warn(
        '`MultiScaleDeformableAttention` in MMCV has been moved to '
        '`mmcv.ops.multi_scale_deform_attn`, please update your MMCV')
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention as MMCV_MSDA

import torch
from torch import nn
from typing import Optional

class MultiScaleDeformableAttention(MMCV_MSDA):
    """warp the MultiScaleDeformableAttention in mmcv.ops"""
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        img2col_step: int = 64,
        dropout: float = 0.1,
        batch_first: bool = False,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            num_levels,
            num_points,
            img2col_step,
            dropout,
            batch_first
        )
        self.im2col_step = img2col_step
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
# ------------------------------------- - ------------------------------------ #


from .dcn_v3 import (
    DCNv3,
    DCNv3Function,
    dcnv3_core_pytorch,
)
from .layer_norm import LayerNorm
from .box_ops import (
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    box_iou,
    generalized_box_iou,
    masks_to_boxes,
)
from .transformer import (
    BaseTransformerLayer,
    TransformerLayerSequence,
)
from .position_embedding import (
    PositionEmbeddingLearned,
    PositionEmbeddingSine,
    get_sine_pos_embed,
)
from .mlp import MLP, FFN
from .attention import (
    MultiheadAttention,
    ConditionalSelfAttention,
    ConditionalCrossAttention,
)
from .conv import (
    ConvNormAct,
    ConvNorm,
)
from .denoising import (
    apply_box_noise,
    apply_label_noise,
    GenerateDNQueries,
)
from .shape_spec import ShapeSpec
