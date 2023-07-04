from .transmotion_tracker import TransMotionTracker

from mmcv.runner import BaseModule

from mmtrack.core import outs2results

import numpy as np
import torch

def single_img_as_batch(img_data):
    for k in img_data.keys():
        if k in ["img", "gt_bboxes", "gt_labels", "gt_instance_ids"]:
            img_data[k] = img_data[k].unsqueeze(0)
        elif k == "img_metas":
            img_data[k] = [img_data[k].data]
    return img_data

def to_target_device(frame, device):
    keys = ["img", "gt_bboxes", "gt_labels", "gt_instance_ids"]
    for k in keys:
        frame[k] = frame[k].to(device)
    return frame

def repeat_gts(frame, n):
    keys =["gt_bboxes", "gt_labels", "gt_instance_ids"]
    for k in keys:
        frame[k] = torch.cat([frame[k]] * n, dim=1)
    return frame

import torch
from torch import nn

from mmdet.datasets.pipelines import Compose

class TransMotion(BaseModule):
    def __init__(
        self,
        detector,
        tracker_cfg,
        num_queries,
        multi_match_k=1,
        freeze_detector=False,
        obj_detect_loss_weight=1.0,
        track_detect_loss_weight=1.0,
    ):
        super().__init__()
        
        self.debug = False
        self.debug_states = {}
        
        self.detector = detector
        self.tracker = TransMotionTracker(**tracker_cfg)
        self.Q = num_queries
        
        self.obj_detect_loss_weight = obj_detect_loss_weight
        self.track_detect_loss_weight = track_detect_loss_weight
        
        for p in self.detector.parameters():
            p.requires_grad_(not freeze_detector)
        
        self._ = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        
        self.multi_match_k = multi_match_k
        
    def freeze_pretrained(self):
        for name, p in self.detector.named_parameters():
            if "track" not in name:
                p.requires_grad_(False)

    def freeze_not_decoder_head(self):
        for p in self.detector.backbone.parameters():
            p.requires_grad_(False)
        for p in self.detector.neck.parameters():
            p.requires_grad_(False)
        for p in self.detector.transformer.encoder.parameters():
            p.requires_grad_(False)
        
    def transform_det_to_track(self, queries, outs):
        outputs_classes, outputs_coords, _, _ = outs
        queries = queries               # L x B x (Q+T) x D
        cls = outputs_classes[-1]       # B x (Q+T) x C
        box = outputs_coords[-1]        # B x (Q+T) x 4
        if queries.shape[2] == self.Q:
            return (queries, cls, box), None
        else:
            return (
                (queries[:, :, :self.Q], cls[:, :self.Q], box[:, :self.Q]), 
                (queries[:, :, self.Q:], cls[:, self.Q:], box[:, self.Q:])
            )
    
    # TODO: batched training
    def forward_train(self, frames=None, *args, **kwargs):
        """Forward function during training."""
        self.tracker.reset()
        
        # single frame training
        if frames is None:
            det_queries, det_references, det_outs, det_losses = self.detector.forward_train(*args, **kwargs)
            return det_losses, None
        
        # video training
        total_det_losses = {}
        total_track_losses = {}
        ## frames are list of dict data with:
        ##   - img: (B, C, H, W)
        ##   - img_metas: list of dict
        ##   - gt_bboxes: list of (N, 4) tensor
        ##   - gt_labels: list of (N, #class) tensor
        ##   - gt_instance_ids: list of (N, 1) tensor
        BS = len(frames[0]["gt_labels"])
        for frame_id, frame_data in enumerate(frames):
            # get track info
            track_embeds = self.tracker.get_track_embeds(frame_id).to(self._.device)
            track_motions = self.tracker.get_track_motions().to(self._.device)
            track_instance_ids = self.tracker.get_track_instance_ids()
            # perform detection
            det_queries, det_references, det_outs, det_losses = self.detector.forward_train(track_embeds=track_embeds, track_motions=track_motions, track_instance_ids=track_instance_ids, **frame_data)
            # perform tracking
            detections, tracks = self.transform_det_to_track(det_queries, det_outs)
            track_losses = self.tracker.track_train(frame_id, detections, tracks, frame_data)
            # gather losses
            total_det_losses.update({f"frame{frame_id}.{key}": val for key, val in det_losses.items()})
            total_track_losses.update({f"frame{frame_id}.{key}": val for key, val in track_losses.items()})
            if self.debug:
                self.debug_states[frame_id] = dict(
                    img=frame_data["img"],
                    det_references=det_references,
                )
        total_losses = {}
        total_obj_det_losses = {key: val for key, val in total_det_losses.items() if "track" not in key}
        total_track_det_losses = {key: val for key, val in total_det_losses.items() if "track" in key}
        # normal object detection losses
        total_losses.update({key: val / len(frames) * self.obj_detect_loss_weight for key, val in total_obj_det_losses.items()})
        # track detection losses (notice that this is calculated per batch)
        total_losses.update({key: val / len(frames) * self.track_detect_loss_weight / BS for key, val in total_track_det_losses.items()})
        # tracker losses (appearance loss)
        total_losses.update({key: val / len(frames) for key, val in total_track_losses.items()})
        return total_losses, frames
        
    def formulate_det_results(self, outs):
        outputs_classes, outputs_coords, _, _ = outs
        cls = outputs_classes[-1].sigmoid()     # B x (Q+T) x C
        box = outputs_coords[-1]                # B x (Q+T) x 4
        assert cls.shape[-1] == 1 and cls.shape[0] == 1
        bboxes = torch.cat([box, cls], dim=-1)[0, :self.Q, :].detach().cpu().numpy()
        labels = np.zeros(self.Q)
        return outs2results(bboxes=bboxes, labels=labels, num_classes=1)
        
    def simple_test(self, img, img_metas, rescale=True, **kwargs):
        """Test without augmentations.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool, optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.

        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """        
        assert self.tracker.num_parallel_batches == 1, "Only support single batch inference"
        
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.tracker.reset()
        
        # get track info
        track_embeds = self.tracker.get_track_embeds(frame_id).to(self._.device)
        track_motions = self.tracker.get_track_motions().to(self._.device)
        track_instance_ids = self.tracker.get_track_instance_ids()
        # perform detection
        det_queries, det_references, det_outs = self.detector.simple_test(img, img_metas, track_embeds=track_embeds, track_motions=track_motions, track_instance_ids=track_instance_ids)
        # perform tracking
        detections, tracks = self.transform_det_to_track(det_queries, det_outs)
        self.tracker.track(frame_id, detections, tracks)
        
        det_result = self.formulate_det_results(det_outs)
        track_result = self.tracker.get_track_results(frame_id)
        
        # convert output
        if rescale:
            H, W, _ = img_metas[0]['ori_shape']
        else:
            H, W, _ = img_metas[0]['img_shape']
        det_bboxes = det_result['bbox_results']
        track_bboxes = track_result['bbox_results']
        C = len(det_bboxes)
        for cid in range(len(det_bboxes)):
            # scale
            det_bboxes[cid][:, :4] = det_bboxes[cid][:, :4] * np.array([W, H, W, H])
            track_bboxes[cid][:, 1:5] = track_bboxes[cid][:, 1:5] * np.array([W, H, W, H])
            # perform cxcywh2xyxy
            det_bboxes[cid][:, 0:2] -= det_bboxes[cid][:, 2:4] / 2
            det_bboxes[cid][:, 2:4] += det_bboxes[cid][:, 0:2]
            track_bboxes[cid][:, 1:3] -= track_bboxes[cid][:, 3:5] / 2
            track_bboxes[cid][:, 3:5] += track_bboxes[cid][:, 1:3]
        
        return dict(
            det_bboxes=det_bboxes,      # list of (N, 5) ndarray [X1, Y1, X2, Y2, score]
            track_bboxes=track_bboxes   # list of (N, 6) ndarray [instance ID, X1, Y1, X2, Y2, score]
        )
        
        