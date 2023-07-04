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


import copy
import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances


class TransMotionDetectorV3(nn.Module):
    """Implement DAB-Deformable-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2201.12329>`_.

    Code is modified from the `official github repo
    <https://github.com/IDEA-opensource/DAB-DETR>`_.

    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 100.
        device (str): Training device. Default: "cuda".
    """

    def __init__(
        self,
        backbone: nn.Module,
        position_embedding: nn.Module,
        neck: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int,
        num_queries: int,
        criterion: nn.Module,
        criterion_track: nn.Module,
        aux_loss: bool = True,
        select_box_nums_for_evaluation: int = 300,
        seperate_det_track_head=True,
        init_cfg: dict = {},
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # define backbone and position embedding module
        self.backbone = backbone
        self.position_embedding = position_embedding

        # define neck module
        self.neck = neck

        # define leanable anchor boxes and learnable tgt embedings.
        # tgt embedings corresponding to content queries in original paper.
        self.num_queries = num_queries
        self.tgt_embed = nn.Embedding(num_queries, embed_dim)
        self.refpoint_embed = nn.Embedding(num_queries, 4)
        # initialize learnable anchor boxes
        nn.init.zeros_(self.tgt_embed.weight)
        nn.init.uniform_(self.refpoint_embed.weight)
        self.refpoint_embed.weight.data[:] = inverse_sigmoid(
            self.refpoint_embed.weight.data[:]
        ).clamp(-3, 3)

        # define transformer module
        self.transformer = transformer

        # define classification head and box head
        raw_class_embed = nn.Linear(embed_dim, num_classes)
        raw_bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes

        # where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.criterion_track = criterion_track

        # init parameters for heads
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        raw_class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(raw_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(raw_bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

        num_pred = transformer.decoder.num_layers
        self.class_embed = nn.ModuleList([copy.deepcopy(raw_class_embed) for i in range(num_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(raw_bbox_embed) for i in range(num_pred)])
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)

        # used for reference point iteration
        self.transformer.decoder.bbox_embed = self.bbox_embed

        self.seperate_det_track_head = seperate_det_track_head
        if seperate_det_track_head:
            num_track_pred = transformer.decoder.num_layers
            nn.init.xavier_uniform_(raw_bbox_embed.layers[-1].weight.data)
            raw_bbox_embed.layers[-1].bias.requires_grad_(False) # REVIEW: hack implementation to prevent bias getting unreasonable large, maybe can replaced by regulation term?
            self.track_class_embed = nn.ModuleList([copy.deepcopy(raw_class_embed) for i in range(num_track_pred)])
            self.track_bbox_embed = nn.ModuleList([copy.deepcopy(raw_bbox_embed) for i in range(num_track_pred)])
            # # REVIEW: this may stablize the training? no, maybe just for detection
            # nn.init.constant_(self.track_bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement and two-stage.
            # The last class_embed and bbox_embed is for region proposal generation
            self.transformer.decoder.track_bbox_embed = self.track_bbox_embed

        # set topk boxes selected for inference
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation
        
        # load pretrained weights
        if init_cfg.get("type", "") == "Pretrained":
            states = torch.load(init_cfg["src"], map_location="cpu")
            missing_keys, unexpected_keys = self.load_state_dict(states["state_dict"], strict=False)
            print(f"missing keys: {missing_keys}")
            print(f"unexpected keys: {unexpected_keys}")

    def init_track_branch_with_det_branch(self):
        """copy detect FFN/Head to track branch"""
        # heads
        for i in range(self.transformer.decoder.num_layers):
            self.track_class_embed[i].load_state_dict(self.class_embed[i].state_dict())
            self.track_bbox_embed[i].load_state_dict(self.bbox_embed[i].state_dict())
        # ffns
        for dlayer in self.transformer.decoder.layers:
            dlayer.track_ffns.load_state_dict(dlayer.ffns.state_dict())

    def forward_train(
        self,
        img,
        img_metas, 
        gt_bboxes,                  # B x N x 4
        gt_labels,                  # B x N
        gt_instance_ids,            # B x N
        track_embeds=None,          # B x T x E
        track_motions=None,         # B x T x 4
        track_instance_ids=None,    # B x T
        gt_bboxes_ignore=None
    ):
        if track_embeds is None or track_motions is None or track_instance_ids is None:
            B, C, H, W = img.shape
            track_embeds = torch.zeros((B, 0, self.embed_dim), device=img.device)
            track_motions = torch.zeros((B, 0, 4), device=img.device)
            track_instance_ids = torch.zeros((B, 0), device=img.device)
        return self.forward(
            img,
            img_metas, 
            track_embeds,           # B x T x E
            track_motions,          # B x T x 4
            track_instance_ids,     # B x T
            gt_bboxes,              # B x N x 4
            gt_labels,              # B x N
            gt_instance_ids,        # B x N
            gt_bboxes_ignore=None
        )
        
    def simple_test(
        self,
        img,
        img_metas, 
        track_embeds,           # B x T x E
        track_motions,          # B x T x 4
        track_instance_ids,     # B x T
    ):
        return self.forward(
            img,
            img_metas, 
            track_embeds,           # B x T x E
            track_motions,          # B x T x 4
            track_instance_ids,     # B x T
            None, None, None, None
        )

    def forward(
        self,
        img,
        img_metas, 
        track_embeds,           # B x T x E
        track_reference_points, # B x T x 4
        track_instance_ids,     # B x T
        gt_bboxes=None,         # list of N x 4
        gt_labels=None,         # list of N
        gt_instance_ids=None,   # list of N
        gt_bboxes_ignore=None        
    ):
        """Forward function of `DAB-Deformable-DETR` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = img
        batch_size, _, H, W = images.shape
        img_masks = images.new_zeros(batch_size, H, W)
        T = track_embeds.shape[1]

        # original features
        features = self.backbone(images)  # output feature dict

        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in DAB-Deformable-DETR
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        # initialize object query embeddings
        query_embeds = self.tgt_embed.weight                # nq, 256
        query_reference_points = self.refpoint_embed.weight # nq, 4

        (
            init_state,
            inter_states,
            init_reference,
            inter_references,
        ) = self.transformer(
            multi_level_feats, multi_level_masks, multi_level_position_embeddings, 
            query_embeds, query_reference_points, track_embeds, track_reference_points
        )

        # Calculate output classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            if self.seperate_det_track_head and T != 0:
                det_states, track_states = torch.split(inter_states[lvl], [self.num_queries, T], dim=1)
                det_cls = self.class_embed[lvl](det_states)
                track_cls = self.track_class_embed[lvl](track_states)
                det_tmp = self.bbox_embed[lvl](det_states)
                track_tmp = self.track_bbox_embed[lvl](track_states)
                outputs_class = torch.cat([det_cls, track_cls], dim=1)
                tmp = torch.cat([det_tmp, track_tmp], dim=1)
            else:
                outputs_class = self.class_embed[lvl](inter_states[lvl])
                tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        raw_outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        raw_outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]
        
        # split track and det outputs
        track_outputs_class = raw_outputs_class[:, :, self.num_queries:, :]
        track_outputs_coord = raw_outputs_coord[:, :, self.num_queries:, :]
        outputs_class = raw_outputs_class[:, :, :self.num_queries, :]
        outputs_coord = raw_outputs_coord[:, :, :self.num_queries, :]

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.training:
            targets = self.prepare_targets(gt_bboxes, gt_labels, gt_instance_ids, img_metas)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            loss_dict.update(self.criterion_track(
                track_outputs_class, 
                track_outputs_coord, 
                track_instance_ids, 
                targets,
            ))
            return (
                torch.cat([init_state.unsqueeze(0), inter_states], dim=0),
                torch.cat([init_reference.unsqueeze(0), raw_outputs_coord], dim=0),
                (raw_outputs_class[[-1]], raw_outputs_coord[[-1]], None, None), 
                loss_dict
            )
        else:
            return (
                torch.cat([init_state.unsqueeze(0), inter_states], dim=0),
                torch.cat([init_reference.unsqueeze(0), raw_outputs_coord], dim=0),
                (raw_outputs_class[[-1]], raw_outputs_coord[[-1]], None, None), 
            )

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # Select top-k confidence boxes for inference
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, gt_bboxes, gt_labels, gt_instance_ids, img_metas):
        targets = []
        for b in range(len(gt_bboxes)):
            h, w, _ = img_metas[b]["img_shape"]
            image_size_xyxy = torch.tensor([[w, h, w, h]], dtype=torch.float, device=gt_bboxes[0].device)
            gt_cls = gt_labels[b]
            gt_box = gt_bboxes[b] / image_size_xyxy
            gt_box = box_xyxy_to_cxcywh(gt_box)
            gt_instance = gt_instance_ids[b]
            targets.append({"labels": gt_cls, "boxes": gt_box, "instance_ids": gt_instance})
        return targets

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]
