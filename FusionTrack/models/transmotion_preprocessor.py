import copy
from typing import Any
import torch

from torchvision import ops

from mmdet.core.bbox import bbox_cxcywh_to_xyxy
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core.bbox.iou_calculators import bbox_overlaps


class FilterBase:
    def __call__(self, dets, tracks):
        """
        Args:
            dets: tuple(Tensor, Tensor, Tensor) [conf, bbox, app]
                - conf: N x C, sigmoid
                - bbox: N x 4, normalized cxcywh
                - app: N x E
            tracks: tuple(Tensor, Tensor, Tensor) [conf, bbox, app]
                - conf: N x C, sigmoid
                - bbox: N x 4, normalized cxcywh
                - app: N x E
        Returns:
            filtered_det_ids: np.ndarray[int]
        """
        pass
    

class NMSFilter(FilterBase):
    """perform nms filtering on detections only"""
    def __init__(self, nms_iou_thres=0.5):
        self.nms_iou_thres = nms_iou_thres
        
    def __call__(self, dets, tracks):
        det_conf, det_bbox, _ = dets
        return ops.nms(bbox_cxcywh_to_xyxy(det_bbox), det_conf.max(dim=-1).values, self.nms_iou_thres)
        
        
class ConfidenceFilter(FilterBase):
    """perform confidence filtering"""
    def __init__(self, conf_thres=0.3):
        self.conf_thres = conf_thres
        
    def __call__(self, dets, tracks):
        return dets[0].max(dim=-1).values > self.conf_thres
 
        
class TrackIoUFilter(FilterBase):
    """filter detections based on iou with tracks"""
    def __init__(self, iou_thres=0.8):
        self.iou_thres = iou_thres
        
    def __call__(self, dets, tracks=None):
        if tracks is None:
            return torch.arange(len(dets[0]))
        else:
            _, det_bbox, _ = dets
            _, track_bbox, _ = tracks
            ious = bbox_overlaps(bbox_cxcywh_to_xyxy(det_bbox), bbox_cxcywh_to_xyxy(track_bbox))
            det_indices = ious.max(dim=-1).values < self.iou_thres
            return det_indices
    
    
FILTERS = {
    "NMSFilter": NMSFilter,
    "ConfidenceFilter": ConfidenceFilter,
    "TrackIoUFilter": TrackIoUFilter,
}


def build_filter(cfg):
    cfg = copy.deepcopy(cfg)
    typename = cfg.pop('type')
    return FILTERS[typename](**cfg)


class DetectPreprocessor:
    def __init__(self, filter_cfgs):
        self.filters = [build_filter(cfg) for cfg in filter_cfgs]
        
    def __call__(self, dets, tracks=None) -> Any:
        conf, bbox, app = dets
        indices = torch.arange(len(conf), device=conf.device)
        for f in self.filters:
            _indices = f((conf[indices], bbox[indices], app[indices]), tracks)
            indices = indices[_indices]
        return conf[indices], bbox[indices], app[indices], indices
    

class TrackPreprocessor:
    def __init__(self, inter_iou_thres=0.8, motion_iou_thres=0.5):
        self.inter_iou_thres = inter_iou_thres
        self.motion_iou_thres = motion_iou_thres
        
    def __call__(self, track_cls, track_box, track_labels, track_motion) -> Any:
        """T x 1, T x 4, T, T x 4"""
        track_box_xyxy = bbox_cxcywh_to_xyxy(track_box)
        track_motion_xyxy = bbox_cxcywh_to_xyxy(track_motion)
        # track nms
        track_conf = track_cls.gather(-1, track_labels.unsqueeze(-1)).squeeze(-1)
        nms_ids = ops.nms(track_box_xyxy, track_conf, self.inter_iou_thres)
        track_box = track_box[nms_ids]
        # motion iou
        T = len(track_box)
        diag_ids = (torch.arange(T), torch.arange(T))
        motion_ious = bbox_overlaps(track_box_xyxy, track_motion_xyxy)
        motion_ious = motion_ious[diag_ids]
        motion_ids = motion_ious >= self.motion_iou_thres
        # merge and return
        return nms_ids[motion_ids]