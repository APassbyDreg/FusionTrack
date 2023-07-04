# ---------------- modified from DAB Deformable DETR in detrex --------------- #

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

from typing import List, Dict

import copy
import warnings

import torch
import torch.nn as nn

from detrex.layers import (
    FFN,
    MLP,
    BaseTransformerLayer,
    MultiheadAttention,
    MultiScaleDeformableAttention,
    TransformerLayerSequence,
    get_sine_pos_embed,
)
from detrex.utils import inverse_sigmoid

from ..utils import TransMotionTransformerDecoderLayer


class TransMotionTransformerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        operation_order: tuple = ("self_attn", "norm", "ffn", "norm"),
        num_layers: int = 6,
        post_norm: bool = False,
        num_feature_levels: int = 4,
    ):
        super(TransMotionTransformerEncoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=MultiScaleDeformableAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    batch_first=True,
                    num_levels=num_feature_levels,
                ),
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    num_fcs=2,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=operation_order,
            ),
            num_layers=num_layers,
        )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm

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
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class TransMotionTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        return_intermediate: bool = True,
        num_feature_levels: int = 4,
        num_det_queries : int = 300,
        mask_track2det: bool = True,
    ):
        super(TransMotionTransformerDecoder, self).__init__(
            transformer_layers=TransMotionTransformerDecoderLayer(
                attn=[
                    MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=True,
                    ),
                    MultiScaleDeformableAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=attn_dropout,
                        batch_first=True,
                        num_levels=num_feature_levels,
                    ),
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                num_det_queries=num_det_queries,
                mask_track2det=mask_track2det,
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate

        self.query_scale = MLP(embed_dim, embed_dim, embed_dim, 2)
        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)

        self.bbox_embed = None          # used per layer
        self.track_bbox_embed = None    # used per layer
        self.class_embed = None         # only used for proposal generation
        self.track_class_embed = None   # not used 
        
        self.num_det_queries = num_det_queries

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
        reference_points=None,  # num_queries, 4. normalized.
        valid_ratios=None,
        **kwargs,
    ):
        output = query
        bs, num_queries, _ = output.size()
        if reference_points.dim() == 2:
            reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)  # bs, num_queries, 4

        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            raw_query_pos = self.ref_point_head(query_sine_embed)
            pos_scale = self.query_scale(output) if layer_idx != 0 else 1
            query_pos = pos_scale * raw_query_pos

            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                **kwargs,
            )

            if self.bbox_embed is not None:
                # ------------- seperate track and detection head ------------ #
                if self.track_bbox_embed is not None and query.shape[1] != self.num_det_queries:
                    T = query.shape[1] - self.num_det_queries
                    out_det, out_track = torch.split(output, [self.num_det_queries, T], dim=1)
                    tmp_det = self.bbox_embed[layer_idx](out_det)
                    tmp_track = self.track_bbox_embed[layer_idx](out_track)
                    tmp = torch.cat([tmp_det, tmp_track], dim=1)
                else:
                    tmp = self.bbox_embed[layer_idx](output)
                # ----------------------------- - ---------------------------- #
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(new_reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        else:
            return output, new_reference_points


class TransMotionTransformer(nn.Module):
    """Transformer module for DAB-Deformable-DETR

    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        as_two_stage (bool): whether to use two-stage transformer. Default False.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 100.
            Only used when as_two_stage is True.
    """

    def __init__(
        self,
        encoder=None,
        decoder=None,
        num_feature_levels=4,
        combine_det_track=False,
    ):
        super(TransMotionTransformer, self).__init__()
        self.combine_det_track = combine_det_track
        self.encoder = encoder
        self.decoder = decoder
        self.num_feature_levels = num_feature_levels

        self.embed_dim = self.encoder.embed_dim

        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.normal_(self.level_embeds)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The ratios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid ratios of feature maps of all levels."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        detect_embeds=None,
        detect_reference_points=None,
        track_embeds=None,
        track_reference_points=None,
        **kwargs,
    ):
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device
        )

        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,  # bs, num_token, num_level, 2
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )

        bs, _, c = memory.shape
        
        detect_reference_points = detect_reference_points.sigmoid().unsqueeze(0).expand(bs, -1, -1)
        detect_embeds = detect_embeds.unsqueeze(0).expand(bs, -1, -1)
        
        # -------------- merge track embeds and reference points ------------- #
        reference_points = torch.cat([detect_reference_points, track_reference_points], dim=1)
        target = torch.cat([detect_embeds, track_embeds], dim=1)
        
        init_reference_points = reference_points
        init_target = target

        if self.combine_det_track or track_embeds.shape[1] == 0:
            inter_states, inter_references = self.decoder(
                query=target,  # bs, num_queries, embed_dims
                key=memory,  # bs, num_tokens, embed_dims
                value=memory,  # bs, num_tokens, embed_dims
                query_pos=None,
                key_padding_mask=mask_flatten,  # bs, num_tokens
                reference_points=reference_points,  # num_queries, 4
                spatial_shapes=spatial_shapes,  # nlvl, 2
                level_start_index=level_start_index,  # nlvl
                valid_ratios=valid_ratios,  # bs, nlvl, 2
                **kwargs,
            )
            return (
                init_target,
                inter_states,
                init_reference_points,
                inter_references,
            )
        else:
            det_states, det_references = self.decoder(
                query=detect_embeds,  # bs, num_queries, embed_dims
                key=memory,  # bs, num_tokens, embed_dims
                value=memory,  # bs, num_tokens, embed_dims
                query_pos=None,
                key_padding_mask=mask_flatten,  # bs, num_tokens
                reference_points=detect_reference_points,  # num_queries, 4
                spatial_shapes=spatial_shapes,  # nlvl, 2
                level_start_index=level_start_index,  # nlvl
                valid_ratios=valid_ratios,  # bs, nlvl, 2
                **kwargs,
            )
            track_states, track_references = self.decoder(
                query=track_embeds,  # bs, num_tracks, embed_dims
                key=memory,  # bs, num_tokens, embed_dims
                value=memory,  # bs, num_tokens, embed_dims
                query_pos=None,
                key_padding_mask=mask_flatten,  # bs, num_tokens
                reference_points=track_reference_points,  # num_tracks, 4
                spatial_shapes=spatial_shapes,  # nlvl, 2
                level_start_index=level_start_index,  # nlvl
                valid_ratios=valid_ratios,  # bs, nlvl, 2
                **kwargs,
            )
            return (
                init_target,
                torch.cat([det_states, track_states], dim=2),
                init_reference_points,
                torch.cat([det_references, track_references], dim=2)
            )
