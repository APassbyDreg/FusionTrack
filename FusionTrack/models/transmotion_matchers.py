import copy

import torch
import numpy as np

from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core import bbox_cxcywh_to_xyxy
from mmdet.core.bbox import bbox_overlaps

from einops import rearrange, reduce, repeat
from scipy.optimize import linear_sum_assignment


def _safe_tondarray(tensor: torch.Tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (list, tuple)):
        return np.array(tensor)
    elif isinstance(tensor, np.ndarray):
        return tensor
    else: 
        raise TypeError(f"Unsupported type {type(tensor)} passed to _safe_tondarray")

def simple_cls_cost(a, b):
    """
    this obeys a simple rule:
    1. detections with large confidence are matched first
    2. weak tracks should be matched harder (because they may detect wrong object)
    """
    assert a.shape[1] == b.shape[1] == 1
    a = repeat(a, "a 1 -> a b", b=b.shape[0])
    b = repeat(b, "b 1 -> a b", a=a.shape[0])
    return 0.5 * (2 - a - b)

def motion_iou_loss(track_box, track_motion):
    T = len(track_box)
    MOTION_COST_START = 0.75    # when iou > MOTION_COST_START, cost is 0
    MOTION_COST_DECAY = 2.00    # defines how fast motion cost decay towards 0 near MOTION_COST_START
    ious = bbox_overlaps(bbox_cxcywh_to_xyxy(track_box), bbox_cxcywh_to_xyxy(track_motion), mode='iou')
    motion_loss = MOTION_COST_START - ious[np.arange(T), np.arange(T)].clamp(max=MOTION_COST_START)
    motion_loss = torch.pow(motion_loss / MOTION_COST_START, MOTION_COST_DECAY)
    return rearrange(motion_loss, "t -> 1 t")


class MatcherBase:
    def match(self, dets, tracks, **kwargs):
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
            matched_det_ids: np.ndarray[int]
            matched_track_ids: np.ndarray[int]
        """
        pass
    
    def merge_costs(self, costs, weights):
        merged_cost = 0
        for k, w in weights.items():
            merged_cost += w * costs[k]
        return merged_cost


class TwoStageMatcher(MatcherBase):
    def __init__(
        self,
        first_match_conf_thres,
        first_match_weights,
        first_match_cost_thres,
        second_match_conf_thres,
        second_match_weights,
        second_match_cost_thres,
    ):
        app_cost_cfg=dict(type='CosineSimilarityCost', weight=1.0)
        iou_cost_cfg=dict(type='IoUCost', iou_mode='giou', weight=1.0)
        reg_cost_cfg=dict(type='LpNormCost', p=1, weight=1.0)
        self.first_match_conf_thres = first_match_conf_thres
        self.first_match_weights = first_match_weights
        self.first_match_cost_thres = first_match_cost_thres
        self.second_match_conf_thres = second_match_conf_thres
        self.second_match_weights = second_match_weights
        self.second_match_cost_thres = second_match_cost_thres
        self.app_cost = build_match_cost(app_cost_cfg)
        self.iou_cost = build_match_cost(iou_cost_cfg)
        self.reg_cost = build_match_cost(reg_cost_cfg)
        self.cls_cost = simple_cls_cost
        self.motion_cost = motion_iou_loss
    
    def match(self, dets, tracks, **kwargs):
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
            matched_det_ids: np.ndarray[int]
            matched_track_ids: np.ndarray[int]
        """
        # ------------------------- extract elements ------------------------- #
        det_cls, det_bbox, det_app = dets
        track_cls, track_bbox, track_app = tracks
        D = len(det_cls)
        T = len(track_cls)
        
        # ---------------------------- calc costs ---------------------------- #
        app_cost = self.app_cost(det_app, track_app)
        iou_cost = self.iou_cost(bbox_cxcywh_to_xyxy(det_bbox), bbox_cxcywh_to_xyxy(track_bbox))
        reg_cost = self.reg_cost(det_bbox, track_bbox)
        cls_cost = self.cls_cost(det_cls, track_cls)
        if "track_motion" in kwargs.keys() and kwargs["track_motion"] is not None:
            motion_cost = self.motion_cost(track_bbox, kwargs["track_motion"])
        else:
            motion_cost = 0
        costs = dict(app=app_cost, iou=iou_cost, cls=cls_cost, reg=reg_cost, motion=motion_cost)
        
        # results
        matched_det_ids = []
        matched_track_ids = []
        
        # -------------------------- first match ------------------------- #
        # filter high score detections
        high_score_det_loc = det_cls.max(dim=-1).values > self.first_match_conf_thres
        high_score_det_ids = np.nonzero(high_score_det_loc.cpu()).squeeze(-1)
        # first match
        first_match_costs = self.merge_costs(costs, self.first_match_weights)
        first_match_costs = first_match_costs[high_score_det_ids, :]
        first_match_costs = _safe_tondarray(first_match_costs)
        raw_matched_det_ids, raw_matched_track_ids = linear_sum_assignment(first_match_costs)
        # filter by thres
        match_cost = first_match_costs[raw_matched_det_ids, raw_matched_track_ids]
        indices = _safe_tondarray(match_cost < self.first_match_cost_thres)
        matched_det_ids.append(high_score_det_ids[raw_matched_det_ids[indices]])
        matched_track_ids.append(raw_matched_track_ids[indices])
        
        # ------------------------- second match ------------------------- #
        # get remain track and detection ids
        if len(matched_track_ids[0]) < T:
            remain_track_ids = torch.tensor(list(set(range(T)) - set(matched_track_ids[0])))
            remain_det_loc = det_cls.max(dim=-1).values > self.second_match_conf_thres
            remain_det_loc[matched_det_ids[0]] = False
            remain_det_ids = torch.nonzero(remain_det_loc).squeeze(-1)
            # second match
            second_match_costs = self.merge_costs(costs, self.second_match_weights)
            second_match_costs = _safe_tondarray(second_match_costs[remain_det_ids, :][:, remain_track_ids])
            raw_matched_det_ids, raw_matched_track_ids = linear_sum_assignment(second_match_costs)
            # filter by thres
            match_cost = second_match_costs[raw_matched_det_ids, raw_matched_track_ids]
            matched_det_ids.append(
                _safe_tondarray(remain_det_ids[raw_matched_det_ids])
            )
            matched_track_ids.append(
                _safe_tondarray(remain_track_ids[raw_matched_track_ids])
            )
            
        # merge results
        matched_det_ids = np.concatenate(matched_det_ids)
        matched_track_ids = np.concatenate(matched_track_ids)
        return matched_det_ids, matched_track_ids, costs


class SingleStageMatcher(MatcherBase):
    def __init__(
        self,
        cost_weights,
        cost_thres,
    ):
        self.cost_weights = cost_weights
        self.cost_thres = cost_thres
        app_cost_cfg=dict(type='CosineSimilarityCost', weight=1.0)
        iou_cost_cfg=dict(type='IoUCost', iou_mode='giou', weight=1.0)
        reg_cost_cfg=dict(type='LpNormCost', p=1, weight=1.0)
        self.app_cost = build_match_cost(app_cost_cfg)
        self.iou_cost = build_match_cost(iou_cost_cfg)
        self.reg_cost = build_match_cost(reg_cost_cfg)
        self.cls_cost = simple_cls_cost
        self.motion_cost = motion_iou_loss
    
    def match(self, dets, tracks, **kwargs):
        # ------------------------- extract elements ------------------------- #
        det_cls, det_bbox, det_app = dets
        track_cls, track_bbox, track_app = tracks
        D = len(det_cls)
        T = len(track_cls)
        
        # ---------------------------- calc costs ---------------------------- #
        app_cost = self.app_cost(det_app, track_app)
        iou_cost = self.iou_cost(bbox_cxcywh_to_xyxy(det_bbox), bbox_cxcywh_to_xyxy(track_bbox))
        reg_cost = self.reg_cost(det_bbox, track_bbox)
        cls_cost = self.cls_cost(det_cls, track_cls)
        if "track_motion" in kwargs.keys() and kwargs["track_motion"] is not None:
            motion_cost = self.motion_cost(track_bbox, kwargs["track_motion"])
        else:
            motion_cost = 0
        raw_costs = dict(app=app_cost, iou=iou_cost, cls=cls_cost, reg=reg_cost, motion=motion_cost)
        
        # ------------------------------- match ------------------------------ #
        costs = self.merge_costs(raw_costs, self.cost_weights)
        costs = _safe_tondarray(costs)
        raw_matched_det_ids, raw_matched_track_ids = linear_sum_assignment(costs)
        # filter by thres
        match_cost = costs[raw_matched_det_ids, raw_matched_track_ids]
        indices = _safe_tondarray(match_cost < self.cost_thres)
        
        return raw_matched_det_ids[indices], raw_matched_track_ids[indices], costs
        

class DoNothingMatcher(MatcherBase):
    """this is only used with TrackIoUFilter preprocessor"""
    def match(self, *args, **kwargs):
        return [], [], None

MATCHERS = {
    "TwoStageMatcher": TwoStageMatcher,
    "SingleStageMatcher": SingleStageMatcher,
    "DoNothingMatcher": DoNothingMatcher,
}

def build_matcher(_cfg):
    cfg = copy.deepcopy(_cfg)
    type_name = cfg.pop('type')
    return MATCHERS[type_name](**cfg)