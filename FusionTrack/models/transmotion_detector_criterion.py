# ---------------- modified from two stage criterion in detrex --------------- #

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

from detrex.modeling.criterion import SetCriterion
from detrex.utils import get_world_size, is_dist_avail_and_initialized


from torch import nn
from einops import rearrange, repeat, reduce
from mmdet.core import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.models.builder import build_loss

class TransMotionDetectorTrackCriterion(nn.Module):
    def __init__(self, loss_cls, loss_box, loss_iou) -> None:
        super().__init__()
        self.loss_cls = build_loss(loss_cls)
        self.loss_box = build_loss(loss_box)
        self.loss_iou = build_loss(loss_iou)
        
        self.debug=False
        self.debug_states = []
        
    def forward(self, track_class, track_coord, track_instance_ids, targets):
        if track_class.shape[2] == 0:
            self.debug_states.append({})
            return {}
        # ------------------------------- basic ------------------------------ #
        device = track_class.device
        T = track_class.shape[1]
        C = track_class.shape[-1]
        BOX_DIM = track_coord.shape[-1]
        BS = len(targets)
        # -------------------------- compute losses -------------------------- #
        losses = {}
        total_gts = 0
        for b, target in enumerate(targets):
            num_gts = len(target["instance_ids"])
            total_gts += num_gts
            # convert gt format
            gt_instance_ids = target["instance_ids"]
            gt_labels = target["labels"]
            gt_bboxes = target["boxes"]
            gts = {
                int(gt_instance_ids[i]): [gt_labels[i], gt_bboxes[i]] 
                for i in range(len(gt_instance_ids))
            }
            # generate targets
            target_cls = []
            target_box = []
            target_box_weights = []
            num_pos = 0
            for instance_id in track_instance_ids[b]:
                instance_id = int(instance_id)
                if instance_id in gts:
                    target_cls.append(gts[instance_id][0])
                    target_box.append(gts[instance_id][1])
                    target_box_weights.append(1.0)
                    num_pos += 1
                else:
                    target_cls.append(C)
                    target_box.append(torch.zeros(BOX_DIM).to(device))
                    target_box_weights.append(0.0)
            target_cls = torch.tensor(target_cls).to(device)                        # T
            cls_weights = torch.ones_like(target_cls).to(device)                    # T
            cls_avg_factor = max(num_pos, 1)
            target_box_weights = torch.tensor(target_box_weights).to(device)        # T
            target_box_weights = repeat(target_box_weights, "t -> t b", b=BOX_DIM)  # T x 4
            target_box = torch.stack(target_box, dim=0).to(device)                  # T x 4 (normalize cxcywh)
            target_box_coords = bbox_cxcywh_to_xyxy(target_box)                     # T x 4 (normalize x1y1x2y2)
            # compute gt weighted losses
            for layer in range(track_class.shape[0]):
                name_prefix = f"d{layer}.track_loss"
                pred_cls = track_class[layer][b]                                # T x C
                pred_box = track_coord[layer][b]                                # T x 4 (normalize cxcywh)
                pred_box_coords = bbox_cxcywh_to_xyxy(pred_box)                 # T x 4 (normalize x1y1x2y2)
                losses[f"{name_prefix}_cls_b{b}"] = self.loss_cls(pred_cls, target_cls, cls_weights, avg_factor=cls_avg_factor) * num_gts
                if num_pos > 0:
                    losses[f"{name_prefix}_iou_b{b}"] = self.loss_iou(pred_box_coords, target_box_coords, target_box_weights, avg_factor=num_pos) * num_gts
                    losses[f"{name_prefix}_box_b{b}"] = self.loss_box(pred_box, target_box, target_box_weights, avg_factor=num_pos) * num_gts
        # normalized losses with total gts
        for loss_name in losses:
            losses[loss_name] /= (max(total_gts, 1) / BS)            
        
        if self.debug:
            self.debug_states.append(dict(
                pred_cls = track_class,
                pred_box = track_coord,
                target_cls = target_cls,
                target_box = target_box
            ))
        return losses
    
    

class TransMotionDetectorDetectCriterion(SetCriterion):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses=["class", "boxes"],
        eos_coef=None,
        loss_class_type="focal_loss",
        alpha: float = 0.25,
        gamma: float = 2,
        two_stage_binary_cls=False,
    ):
        super().__init__(
            num_classes, matcher, weight_dict, losses, eos_coef, loss_class_type, alpha, gamma
        )
        self.two_stage_binary_cls = two_stage_binary_cls

    def forward(self, outputs, targets, return_indices=False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # for two stage
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            if self.two_stage_binary_cls:
                for bt in targets:
                    bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, targets)
            for loss in self.losses:
                if loss == "masks":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
