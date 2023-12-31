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

import torch
import torch.nn as nn

from detrex.layers import (
    FFN,
    MLP,
    BaseTransformerLayer,
    ConditionalCrossAttention,
    ConditionalSelfAttention,
    MultiheadAttention,
    TransformerLayerSequence,
    get_sine_pos_embed,
)
from detrex.utils import inverse_sigmoid


class DabDetrTransformerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.1,
        activation: nn.Module = nn.PReLU(),
        post_norm: bool = False,
        num_layers: int = 6,
        batch_first: bool = False,
    ):
        super(DabDetrTransformerEncoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop=attn_dropout,
                    batch_first=batch_first,
                ),
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    ffn_drop=ffn_dropout,
                    activation=activation,
                ),
                norm=nn.LayerNorm(normalized_shape=embed_dim),
                operation_order=("self_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):

        for layer in self.layers:
            position_scales = self.query_scale(query)
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos * position_scales,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class DabDetrTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.0,
        activation: nn.Module = nn.PReLU(),
        num_layers: int = None,
        modulate_hw_attn: bool = True,
        batch_first: bool = False,
        post_norm: bool = True,
        return_intermediate: bool = True,
    ):
        super(DabDetrTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=[
                    ConditionalSelfAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=batch_first,
                    ),
                    ConditionalCrossAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=batch_first,
                    ),
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    ffn_drop=ffn_dropout,
                    activation=activation,
                ),
                norm=nn.LayerNorm(
                    normalized_shape=embed_dim,
                ),
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate
        self.embed_dim = self.layers[0].embed_dim
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)

        self.bbox_embed = None

        if modulate_hw_attn:
            self.ref_anchor_head = MLP(self.embed_dim, self.embed_dim, 2, 2)
        self.modulate_hw_attn = modulate_hw_attn

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

        for idx in range(num_layers - 1):
            self.layers[idx + 1].attentions[1].query_pos_proj = None

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        anchor_box_embed=None,
        **kwargs,
    ):
        intermediate = []

        reference_boxes = anchor_box_embed.sigmoid()
        intermediate_ref_boxes = [reference_boxes]

        for idx, layer in enumerate(self.layers):
            obj_center = reference_boxes[..., : self.embed_dim]
            query_sine_embed = get_sine_pos_embed(obj_center)
            query_pos = self.ref_point_head(query_sine_embed)

            # do not apply transform in position in the first decoder layer
            if idx == 0:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)

            # apply position transform
            query_sine_embed = query_sine_embed[..., : self.embed_dim] * position_transform

            if self.modulate_hw_attn:
                ref_hw_cond = self.ref_anchor_head(query).sigmoid()
                query_sine_embed[..., self.embed_dim // 2 :] *= (
                    ref_hw_cond[..., 0] / obj_center[..., 2]
                ).unsqueeze(-1)
                query_sine_embed[..., : self.embed_dim // 2] *= (
                    ref_hw_cond[..., 1] / obj_center[..., 3]
                ).unsqueeze(-1)

            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                is_first_layer=(idx == 0),
                **kwargs,
            )

            # update anchor boxes after each decoder layer using shared box head.
            if self.bbox_embed is not None:
                # predict offsets and added to the input normalized anchor boxes.
                offsets = self.bbox_embed(query)
                offsets[..., : self.embed_dim] += inverse_sigmoid(reference_boxes)
                new_reference_boxes = offsets[..., : self.embed_dim].sigmoid()

                if idx != self.num_layers - 1:
                    intermediate_ref_boxes.append(new_reference_boxes)
                reference_boxes = new_reference_boxes.detach()

            if self.return_intermediate:
                if self.post_norm_layer is not None:
                    intermediate.append(self.post_norm_layer(query))
                else:
                    intermediate.append(query)

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(query)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(intermediate_ref_boxes).transpose(1, 2),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    reference_boxes.unsqueeze(0).transpose(1, 2),
                ]

        return query.unsqueeze(0)


class DabDetrTransformer(nn.Module):
    def __init__(self, encoder=None, decoder=None, num_patterns=0):
        super(DabDetrTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dim = self.encoder.embed_dim

        # using patterns designed as AnchorDETR
        assert isinstance(num_patterns, int), "num_patterns should be int but got {}".format(type(num_patterns))
        self.num_patterns = num_patterns
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, self.embed_dim)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask, anchor_box_embed, pos_embed):
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(2, 0, 1)  # (c, bs, num_queries)
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        anchor_box_embed = anchor_box_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.view(bs, -1)
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask,
        )
        num_queries = anchor_box_embed.shape[0]


        if self.num_patterns == 0:
            target = torch.zeros(num_queries, bs, self.embed_dim, device=anchor_box_embed.device)
        else:
            target = self.patterns.weight[:, None, None, :].repeat(1, num_queries, bs, 1).flatten(0, 1)
            anchor_box_embed = anchor_box_embed.repeat(self.num_patterns, 1, 1)

        hidden_state, reference_boxes = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            anchor_box_embed=anchor_box_embed,
        )

        return hidden_state, reference_boxes
