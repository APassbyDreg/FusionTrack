from typing import Optional
import torch
import torch.nn as nn

from torchvision import ops

import numpy as np

from scipy.optimize import linear_sum_assignment

from mmcv.cnn import build_activation_layer, build_norm_layer

from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core import build_assigner, build_sampler, bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.core.bbox.iou_calculators import bbox_overlaps

from mmtrack.core import outs2results

from mmdet.models.builder import build_loss

from mmtrack.models.motion.kalman_filter import KalmanFilterCXCYWH

from einops import rearrange, reduce

from mmcv.runner import BaseModule

from detrex.layers import FFN, MLP

from .transmotion_matchers import build_matcher, simple_cls_cost
from .transmotion_preprocessor import DetectPreprocessor, TrackPreprocessor
from .utils import ScoreEmbed, build_fuser

import math

def next_pow_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

def lerp(x, y, t):
    return x * (1 - t) + y * t


class TransMotionConverter(BaseModule):
    """MLP to convert appearance feature to new query"""
    def __init__(
        self,
        d_query,
        d_ffn,
        d_out,
        num_layers=2,
        act=dict(type="GELU"),
        norm=dict(type="LN"),
        dropout=0.1,
    ):
        super().__init__()
        assert num_layers >= 2
        self.proj_in = nn.Sequential(
            nn.Linear(d_query, d_ffn),
            build_activation_layer(act),
            build_norm_layer(norm, d_ffn)[1]
        )
        intermidiates = []
        for _ in range(num_layers - 2):
            intermidiates.append(nn.Sequential(
                nn.Linear(d_ffn, d_ffn),
                build_activation_layer(act),
                build_norm_layer(norm, d_ffn)[1],
                nn.Dropout(dropout),
            ))
        self.intermidiates = nn.ModuleList(intermidiates)
        self.proj_out = nn.Sequential(
            nn.Linear(d_ffn, d_out),
            build_activation_layer(act),
            build_norm_layer(norm, d_out)[1],
            nn.Dropout(dropout),
        )
    
    def forward(self, q):
        q = self.proj_in(q)
        for inter in self.intermidiates:
            q = inter(q) + q
        q = self.proj_out(q)
        return q


class Tracklet:
    FAKE_INSTANCE_ID = -2
    def __init__(self, idx, frame, query, box, conf, label, instance=-1) -> None:
        self.idx = int(idx)
        self.latest_query = query
        self.queries = {frame: query}
        self.boxes = {frame: box.cpu().tolist()}
        self.confidences = {frame: float(conf)}
        self.last_update = int(frame)
        self.label = int(label)
        self.instance = int(instance)
        self.frames = set([frame])
        # motion part
        self.kf = KalmanFilterCXCYWH()
        box_kf = box.cpu().detach().numpy()
        self.kf_mean, self.kf_cov = self.kf.initiate(box_kf)
        
    def set_state(self, frame, state, value):
        name = f"{state}_frame{frame}"
        setattr(self, name, value)
        
    def get_state(self, frame, state):
        name = f"{state}_frame{frame}"
        return getattr(self, name, None)
        
    def update(self, frame, query, box, conf):
        self.confidences[frame] = float(conf)
        self.latest_query = query
        self.queries[frame] = query
        self.boxes[frame] = box.cpu().tolist()
        self.last_update = frame
        self.frames.add(frame)
        # motion part
        box_kf = box.cpu().detach().numpy()
        self.kf_mean, self.kf_cov = self.kf.update(self.kf_mean, self.kf_cov, box_kf)

    def predict(self, frame):
        """should run per frame"""
        if self.last_update != frame:
            self.kf_mean[6] = 0
            self.kf_mean[7] = 0
        self.kf_mean, self.kf_cov = self.kf.predict(self.kf_mean, self.kf_cov)


class TransMotionTracker(nn.Module):
    def __init__(
        self,
        fuser_cfg,
        converter_cfg,
        reid_cfg,
        # ------------------------- matching configs ------------------------- #
        det_preprocess_cfg=dict(filter_cfgs=list()),
        track_preprocess_cfg=dict(motion_iou_thres=0.5),
        matcher_cfg=dict(),
        keep_frame=5,
        track_update_rate=0.5,              # used to update tracklet states
        new_track_conf_thres=[0.5, 0.75],   # used to create new tracklets (first frame / others)
        continue_track_conf_thres=0.5,      # used to continue tracks using unmatched tracks
        # --------------------------- training only -------------------------- #
        app_loss=None,
        train_cfg=None,
        debug=False,
        num_parallel_batches=1,
        # -------------------------- extra featrues -------------------------- #
        use_score_embed=False,
        use_time_embed=False
    ) -> None:
        super().__init__()
        self.debug = debug
        self.reset_debug_states()
        
        # implementation of parallel training
        # tracks are padded to the same length for all batches
        self.num_parallel_batches = num_parallel_batches
        self.out_query_dim = converter_cfg["d_out"]
        
        self.fuser_det = build_fuser(fuser_cfg)
        self.fuser_track = build_fuser(fuser_cfg)
        self.converter = TransMotionConverter(**converter_cfg)
        
        self.det_reid_proj = FFN(**reid_cfg)
        self.track_reid_proj = FFN(**reid_cfg)
        
        self.preprocess_det = DetectPreprocessor(**det_preprocess_cfg)
        self.preprocess_track = TrackPreprocessor(**track_preprocess_cfg)
        self.matcher = build_matcher(matcher_cfg)
        self.keep_frame = keep_frame
        self.track_update_rate = track_update_rate
        self.new_track_conf_thres = new_track_conf_thres
        self.continue_track_conf_thres = continue_track_conf_thres
        
        self.tracks = [{} for _ in range(self.num_parallel_batches)]
        self.terminated_tracks = []
        self.num_ids = [0]
        
        try:
            self.app_cost = self.matcher.app_cost
            self.reg_cost = self.matcher.reg_cost
            self.iou_cost = self.matcher.iou_cost
            self.cls_cost = self.matcher.cls_cost
            self.motion_cost = self.matcher.motion_cost
        except:
            import warnings
            warnings.warn("Matcher does not have all cost functions. Please ensure this model is inference only.")
        
        self.use_score_embed = use_score_embed
        self.use_time_embed = use_time_embed
        if use_score_embed:
            self.score_embed = ScoreEmbed(self.out_query_dim)
        if use_time_embed:
            # NOTE: during training, update this value and copy last time embeds to new time embeds when a new time is seen
            # NOTE: during inference, clamp dt to max_time_seen
            self.max_time_seen = nn.Parameter(torch.tensor(1), requires_grad=False) 
            self.time_embed = nn.Embedding(keep_frame, self.out_query_dim) 
        
        if train_cfg is not None:
            self.assigner = build_assigner(train_cfg['assigner'])
            self.sampler = build_sampler(dict(type='PseudoSampler'))
        if app_loss is not None:
            self.need_loss = True
            self.app_loss = build_loss(app_loss)
        else:
            self.need_loss = False
        
        self.reset()
        self.reset_debug_states()
    
    def reset_debug_states(self):
        # min / max / sum / sqr sum / count
        self.match_cost_info = {
            "app": [float("inf"), float("-inf"), 0, 0, 0],
            "iou": [float("inf"), float("-inf"), 0, 0, 0],
            "cls": [float("inf"), float("-inf"), 0, 0, 0],
        }
        self.unmatch_cost_info = {
            "app": [float("inf"), float("-inf"), 0, 0, 0],
            "iou": [float("inf"), float("-inf"), 0, 0, 0],
            "cls": [float("inf"), float("-inf"), 0, 0, 0],
        }
        # img path / det box / det conf / track box / track conf
        self.records = []
        # cost / record ID / det ID / track ID
        self.matched_costs = []
        self.unmatched_costs = []
        # for few shot debugging
        self.debug_states = {}
    
    def reset(self):
        """reset all states of a tracker"""
        self.tracks = [{} for _ in range(self.num_parallel_batches)]
        self.terminated_tracks = [[] for _ in range(self.num_parallel_batches)]
        self.num_ids = [0 for _ in range(self.num_parallel_batches)]
        if self.debug:
            self.debug_states = {}
        
    def terminate_tracks(self, frame_id):
        """terminate old tracks for all batches"""
        for batch in range(self.num_parallel_batches):
            # terminate tracks
            keep_tracks = []
            terminate_tracks = []
            for track in self.tracks[batch].values():
                if frame_id - track.last_update > self.keep_frame:
                    terminate_tracks.append(track)
                else:
                    keep_tracks.append(track)
            self.tracks[batch] = {t.idx: t for t in keep_tracks}
            self.terminated_tracks[batch].extend(terminate_tracks)
            
    def update_tracks(self, frame_id):
        """update track states and terminate old tracks for all batches"""
        for batch in range(self.num_parallel_batches):
            # update kf
            for track in self.tracks[batch].values():
                track.predict(frame_id)
            # if a frame is missing use motion to continue track state
            for track in self.tracks[batch].values():
                if track.last_update < frame_id:
                    track.frames.add(frame_id)
                    track.boxes[frame_id] = track.kf_mean[:4].tolist()
                    track.confidences[frame_id] = track.confidences[frame_id - 1] / 1.6
    
    def count_fake_tracks(self, batch=0):
        """get fake track count of specific batch"""
        cnt = 0
        for t in self.tracks[batch].values():
            cnt += 1 if t.instance == Tracklet.FAKE_INSTANCE_ID else 0
        return cnt
    
    def get_track_instance_ids(self, batch=None):
        """return track instance ids [B x #Track]"""
        if batch is None:
            if len(self.tracks[0]) == 0:
                return torch.zeros(self.num_parallel_batches, 0)
            result = []
            for batch in range(self.num_parallel_batches):
                ids = np.array([t.idx for t in self.tracks[batch].values()])
                instances = torch.tensor([t.instance for t in self.tracks[batch].values()])
                ids = np.argsort(ids)
                result.append(instances[ids])
            return torch.stack(result, dim=0)
        else:
            if len(self.tracks[batch]) == 0:
                return torch.zeros(0)
            ids = np.array([t.idx for t in self.tracks[batch].values()])
            instances = torch.tensor([t.instance for t in self.tracks[batch].values()])
            ids = np.argsort(ids)
            return instances[ids]
    
    def get_track_embeds(self, frame_id):
        """return track embeddings [B x #Track x D]"""
        if len(self.tracks[0]) == 0:
            return torch.zeros(self.num_parallel_batches, 0, self.out_query_dim)
        batched_queries = []
        batched_scores = []
        batched_dts = []
        for batch in range(self.num_parallel_batches):
            # reorder
            ids = np.array([t.idx for t in self.tracks[batch].values()])
            ids = np.argsort(ids)
            # features
            queries = torch.stack([t.latest_query for t in self.tracks[batch].values()], dim=0)
            batched_queries.append(queries[ids])
            device = queries.device
            # scores
            if self.use_score_embed:
                scores = torch.tensor([t.confidences[t.last_update] for t in self.tracks[batch].values()]).unsqueeze(-1).detach()
                batched_scores.append(scores[ids])
            # delta time
            if self.use_time_embed:
                dts = torch.tensor([frame_id - t.last_update - 1 for t in self.tracks[batch].values()])
                if self.training:
                    maxdt = torch.max(dts)
                    if maxdt > self.max_time_seen:
                        self.max_time_seen += 1
                        self.time_embed.weight.data[maxdt] = self.time_embed.weight.data[maxdt - 1]
                else:
                    dts = torch.clamp(dts.to(device), max=self.max_time_seen-1)
                batched_dts.append(dts[ids])
        batched_queries = torch.stack(batched_queries, dim=0)               # B x #Track x D
        if self.use_score_embed:
            batched_scores = torch.stack(batched_scores, dim=0).to(device)  # B x #Track x 1
            batched_queries = batched_queries + self.score_embed(batched_scores)
        if self.use_time_embed:
            batched_dts = torch.stack(batched_dts, dim=0).to(device)        # B x #Track
            batched_queries = batched_queries + self.time_embed(batched_dts)
        return self.converter(batched_queries)
    
    def get_track_motions(self):
        """return track motions [B x #Track x 4], inference only"""
        if len(self.tracks[0]) == 0:
            return torch.zeros(self.num_parallel_batches, 0, 4)
        device = next(self.parameters()).device
        result = []
        for batch in range(self.num_parallel_batches):
            ids = np.array([t.idx for t in self.tracks[batch].values()])
            motions = torch.stack([torch.tensor(t.kf_mean[:4]) for t in self.tracks[batch].values()], dim=0)
            ids = np.argsort(ids)
            result.append(motions[ids])
        return torch.stack(result, dim=0).float().to(device)
    
    def get_track_labels(self):
        """return track instance ids [B x #Track], inference only"""
        if len(self.tracks[0]) == 0:
            return torch.zeros(self.num_parallel_batches, 0)
        device = next(self.parameters()).device
        result = []
        for batch in range(self.num_parallel_batches):
            ids = np.array([t.idx for t in self.tracks[batch].values()])
            labels = torch.tensor([t.label for t in self.tracks[batch].values()])
            ids = np.argsort(ids)
            result.append(labels[ids])
        return torch.stack(result, dim=0).to(device)
    
    def track_train(self, frame_id, detections, tracks=None, frame_data=None):
        """_summary_

        Args:
            frame_id: int
            detections: tuple([#Layer x B x Q x D], [B x Q x #Class], [B x Q x #Box])
            tracks: tuple([#Layer x B x T x D], [B x T x #Class], [B x T x #Box])
            frame_data: dict
        """
        if self.debug:
            self.debug_states[frame_id] = {
                "detections": detections,
                "tracks": tracks,
            }
            
        device = detections[0].device
        L, B, Q, E = detections[0].shape
        T = tracks[0].shape[2] if tracks is not None else 0
        
        # appearance
        det_feats, det_cls, det_box = detections
        det_feats = det_feats
        det_cls = det_cls.sigmoid().detach()
        det_box = det_box.detach()
        det_app = self.fuser_det(det_feats, det_cls.max(dim=-1, keepdim=True).values) # detach detection branch to avoid backprop
        
        if tracks is None:
            track_feats, track_cls, track_box, track_app = None, None, None, None
        else:
            track_feats, track_cls, track_box = tracks
            track_feats = track_feats
            track_cls = track_cls.sigmoid().detach()
            track_box = track_box.detach()
            track_app = self.fuser_track(track_feats, track_cls) # NOTE: only works with one class
            track_cls = track_cls.sigmoid()
        
        losses = {}
        remain_ids = []
        total_matched_cnt = 0
        for batch in range(B):
            # match detections to GT
            assign_result = self.assigner.assign(
                det_box[batch], 
                det_cls[batch], 
                frame_data['gt_bboxes'][batch], 
                frame_data['gt_labels'][batch], 
                frame_data['img_metas'][batch], 
                None
            )
            sampling_result = self.sampler.sample(
                assign_result, 
                det_box[batch], 
                frame_data['gt_bboxes'][batch]
            )
            # instance ids for det
            NULL_INSTANCE_ID = -1
            det_matched_instances = torch.ones(Q, dtype=torch.long).to(device) * NULL_INSTANCE_ID
            det_matched_instances[sampling_result.pos_inds] = frame_data['gt_instance_ids'][batch][sampling_result.pos_assigned_gt_inds]
            # instance ids for tracks
            track_instance_ids = self.get_track_instance_ids(batch).to(device)
            # generate det/track match & appearance cost target & masks
            matched_det_inds = []
            matched_track_inds = []
            cost_mask = torch.zeros((Q, T), dtype=torch.float32).to(device)
            app_sim_target = torch.zeros((Q, T), dtype=torch.float32).to(device)
            instance2det = {}
            unmatch_weight = 1 / Q
            for i, instance_id in enumerate(det_matched_instances):
                instance_id = int(instance_id)
                if instance_id != NULL_INSTANCE_ID:
                    instance2det[instance_id] = i
                    cost_mask[i, :] = unmatch_weight
            for i, instance_id in enumerate(track_instance_ids):
                instance_id = int(instance_id)
                cost_mask[:, i] = unmatch_weight
                if instance_id in instance2det:
                    matched_det_inds.append(instance2det[instance_id])
                    matched_track_inds.append(i)
                    app_sim_target[instance2det[instance_id], i] = 1
                    cost_mask[instance2det[instance_id], :] = -1
                    cost_mask[instance2det[instance_id], i] = 1
            cost_mask[cost_mask < 0] = 1 / max(1, len(matched_det_inds))
            cost_mask /= cost_mask.sum()
            total_matched_cnt += len(matched_det_inds)
            
            if tracks is not None:
                # compute reid cost and loss (detached from main branch)
                det_app_reid = self.det_reid_proj(det_app[batch].detach())
                track_app_reid = self.track_reid_proj(track_app[batch].detach())
                app_cost = self.app_cost(det_app_reid, track_app_reid)
                if self.debug:
                    iou_cost = self.iou_cost(bbox_cxcywh_to_xyxy(det_box[batch]), bbox_cxcywh_to_xyxy(track_box[batch]))
                    cls_cost = self.cls_cost(det_cls[batch], track_cls[batch])
                    reg_cost = self.reg_cost(det_box[batch], track_box[batch])
                    # more statistics
                    record_id = len(self.records)
                    self.records.append(dict(
                        # img=frame_data['img'][batch].detach().cpu().numpy(),
                        img_path=frame_data['img_metas'][batch]['filename'],
                        det_box=det_box[batch].detach().cpu().numpy(),
                        track_box=track_box[batch].detach().cpu().numpy(),
                        det_cls=det_cls[batch].detach().cpu().numpy(),
                        track_cls=track_cls[batch].detach().cpu().numpy(),
                    ))
                    from einops import repeat
                    det_idx = np.arange(Q)
                    track_idx = np.arange(T)
                    det_idx = repeat(det_idx, 'q -> q t', t=T)
                    track_idx = repeat(track_idx, 't -> q t', q=Q)
                    record_idx = np.ones([Q, T]) * record_id
                    tmp = app_cost + iou_cost + cls_cost
                    tmp[matched_det_inds, matched_track_inds] = float("-inf")
                    unmatched_cost_locs = np.unravel_index(tmp.view(-1).sort().indices[len(matched_det_inds):].cpu(), [Q, T])
                    cost_info_group = np.stack(
                        [
                            record_idx, det_idx, track_idx,
                            app_cost.detach().cpu().numpy(),
                            iou_cost.detach().cpu().numpy(), 
                            cls_cost.detach().cpu().numpy(),
                            reg_cost.detach().cpu().numpy(),
                        ], 
                        axis=-1
                    )
                    self.matched_costs.extend(cost_info_group[matched_det_inds, matched_track_inds].tolist())
                    self.unmatched_costs.extend(cost_info_group[unmatched_cost_locs].tolist())
                    
                
                if self.need_loss:
                    # compute appearance loss
                    app_sim = 1 - app_cost / self.app_cost.weight * 2
                    app_loss = self.app_loss(app_sim, app_sim_target, cost_mask, avg_factor=1.0)
                    losses[f"app_loss_{batch}"] = app_loss * len(matched_det_inds)
            
                # use match result to update track
                track_ids = np.array(sorted([t.idx for t in self.tracks[batch].values()]))
                for track_idx, det_idx in zip(matched_track_inds, matched_det_inds):
                    tid = track_ids[track_idx]
                    label = self.tracks[batch][tid].label
                    conf_det = det_cls[batch][det_idx][label]
                    conf_track = track_cls[batch][track_idx][label]
                    interp = conf_det / (conf_det + conf_track)
                    query = track_app[batch][track_idx] # use track appearance directly as query, we assume detection infomation is fused in previous steps
                    box = lerp(track_box[batch][track_idx], det_box[batch][det_idx], interp)
                    conf = lerp(track_cls[batch][track_idx][label], det_cls[batch][det_idx][label], interp)
                    self.tracks[batch][tid].update(frame_id, query, box, conf)
                
                # take unmatched dets out 
                unmatched_det_ids = np.array(set(range(Q)) - set(matched_det_inds))
            else:
                unmatched_det_ids = np.arange(Q)
            
            # use match result to initialize new tracks
            # get unmatched dets
            inds = set(sampling_result.pos_inds.tolist()) & set(unmatched_det_ids.tolist())
            # init tracks
            for det_id in inds:
                tid = self.num_ids[batch]
                query = det_app[batch][det_id]
                box = det_box[batch][det_id]
                label = det_cls[batch][det_id].argmax()
                conf = det_cls[batch][det_id][label]
                self.tracks[batch][self.num_ids[batch]] = Tracklet(
                    tid, frame_id,
                    query, box, conf, label, 
                    det_matched_instances[det_id]
                )
                self.num_ids[batch] += 1
            
            # negative samples
            remain_ids.append(set(unmatched_det_ids.tolist()) - set(sampling_result.pos_inds.tolist()))
            
        # terminate tracks
        self.terminate_tracks(frame_id)    
        
        # pad tracks tgt_pad to align
        tgt_pad = 0
        for batch in range(self.num_parallel_batches):
            num_tracks = len(self.tracks[batch])
            num_fake_tracks = self.count_fake_tracks(batch)
            num_real_tracks = num_tracks - num_fake_tracks
            tgt_pad = max(num_real_tracks, tgt_pad)
        tgt_pad = int(tgt_pad * 1.1)
        for batch in range(self.num_parallel_batches):
            # remove or add some fake tracks to align
            delta = tgt_pad - len(self.tracks[batch])
            if delta <= 0:
                rm_tids = []
                for tid in self.tracks[batch].keys():
                    if len(rm_tids) == abs(delta):
                        break
                    if self.tracks[batch][tid].instance == Tracklet.FAKE_INSTANCE_ID:
                        rm_tids.append(tid)
                for tid in rm_tids:
                    self.tracks[batch].pop(tid)
            else:
                num_available_ids = len(remain_ids[batch])
                if num_available_ids > 0:
                    # utilize remain dets
                    inds = np.random.choice(list(remain_ids[batch]), min(delta, num_available_ids))
                    for det_id in inds:
                        tid = self.num_ids[batch]
                        query = det_app[batch][det_id]
                        box = det_box[batch][det_id]
                        label = det_cls[batch][det_id].argmax()
                        conf = det_cls[batch][det_id][label]
                        self.tracks[batch][self.num_ids[batch]] = Tracklet(
                            tid, frame_id,
                            query, box, conf, label, 
                            Tracklet.FAKE_INSTANCE_ID
                        )
                        self.num_ids[batch] += 1
                    delta -= len(inds)
                if delta > 0:
                    # generate random fake tracks when remain dets are not enough
                    for _ in range(delta):
                        tid = self.num_ids[batch]
                        query = torch.randn_like(det_app[batch][0])
                        box = torch.rand_like(det_box[batch][0])
                        label = 0
                        conf = 0
                        self.tracks[batch][self.num_ids[batch]] = Tracklet(
                            tid, frame_id,
                            query, box, conf, label, 
                            Tracklet.FAKE_INSTANCE_ID
                        )
                        self.num_ids[batch] += 1
                
        # update tracks
        self.update_tracks(frame_id)
            
        return { k: v / max(1.0, total_matched_cnt) for k, v in losses.items() }
     
    def track(self, frame_id, detections, tracks=None, frame_data=None):
        """
        Args:
            detections: tuple([#Layer x B x Q x D], [B x Q x #Class], [B x Q x #Box])
            tracks: tuple([#Layer x B x T x D], [B x T x #Class], [B x T x #Box])
        """
        if self.debug:
            self.debug_states[frame_id] = {
                "detections": detections,
                "tracks": tracks,
            }
        
        device = detections[0].device
        B = detections[0].shape[1]
        T = tracks[0].shape[1] if tracks is not None else 0
        assert B == 1, "Only support batch size 1"
        
        # appearance
        det_feats, det_cls, det_box = detections
        det_app = self.fuser_det(det_feats, det_cls.max(dim=-1, keepdim=True).values)[0]
        det_cls = det_cls[0].sigmoid()
        det_box = det_box[0]
        
        if tracks is None:
            det_cls, det_box, det_app, valid_indices = self.preprocess_det((det_cls, det_box, det_app), None)
            Q = len(det_cls)
            matched_det_ids = []
        else:
            track_ids = torch.tensor(sorted([t.idx for t in self.tracks[0].values()]))
            track_feats, track_cls, track_box = tracks
            track_motion = self.get_track_motions()[0]
            track_labels = self.get_track_labels()[0]
            indices = self.preprocess_track(track_cls[0], track_box[0], track_labels, track_motion).detach().cpu()
            track_motion = track_motion[indices]
            track_feats = track_feats[:, :, indices]
            track_cls = track_cls[:, indices]
            track_box = track_box[:, indices]
            track_ids = track_ids[indices].tolist()
            track_app = self.fuser_track(track_feats, track_cls)[0] # NOTE: only works with one class
            track_cls = track_cls[0].sigmoid()
            track_box = track_box[0]
            
            det_cls, det_box, det_app, valid_indices = self.preprocess_det(
                dets=(det_cls, det_box, det_app), 
                tracks=(track_cls, track_box, track_app)
            )
            Q = len(det_cls)
            T = len(track_cls)
            
            # ----------------------- perform matching ----------------------- #
            det_app_reid = self.det_reid_proj(det_app)
            track_app_reid = self.track_reid_proj(track_app)
            matched_det_ids, matched_track_ids, costs = self.matcher.match(
                dets=(det_cls, det_box, det_app_reid),
                tracks=(track_cls, track_box, track_app_reid),
                track_motion=track_motion
            )
            if self.debug:
                self.debug_states[frame_id]["costs"] = costs
                self.debug_states[frame_id]["matched_det_ids"] = matched_det_ids
                self.debug_states[frame_id]["matched_track_ids"] = matched_track_ids
                self.debug_states[frame_id]["processed_dets"] = (det_cls, det_box)
                self.debug_states[frame_id]["processed_tracks"] = (track_cls, track_box, track_ids, indices)
                self.debug_states[frame_id]["track_motion"] = track_motion
        
            # ------------------------- update tracks ------------------------ #
            # update matched tracks
            for det_idx, track_idx in zip(matched_det_ids, matched_track_ids):
                tid = track_ids[track_idx]
                label = self.tracks[0][tid].label
                conf_det = det_cls[det_idx][label]
                conf_track = track_cls[track_idx][label]
                interp = conf_det / (conf_det + conf_track)
                query = track_app[track_idx] # use track appearance directly as query, we assume detection infomation is fused in previous steps
                box = lerp(track_box[track_idx], det_box[det_idx], interp)
                conf = lerp(track_cls[track_idx][label], det_cls[det_idx][label], interp) 
                self.tracks[0][tid].update(frame_id, query, box, conf)
            
            # try to update unmatched tracks tracks
            unmatched_track_ids = np.array(list(set(range(T)) - set(matched_track_ids)))
            if len(unmatched_track_ids) > 0:
                unmatched_track_labels = [self.tracks[0][track_ids[i]].label for i in unmatched_track_ids]
                unmatched_track_confs = track_cls[unmatched_track_ids, unmatched_track_labels]
                indices = (unmatched_track_confs > self.continue_track_conf_thres).cpu().numpy()
                for track_idx in unmatched_track_ids[indices]:
                    tid = track_ids[track_idx]
                    label = self.tracks[0][tid].label
                    query = track_app[track_idx]
                    box = track_box[track_idx]
                    conf = track_cls[track_idx][label]
                    self.tracks[0][tid].update(frame_id, query, box, conf)

        # ----------------------- initialize new tracks ---------------------- #
        # take unmatched dets out
        unmatched_det_ids = np.array(list(set(range(Q)) - set(matched_det_ids)))
        
        # use confidences to initialize new tracks
        # filter unmatched dets by conf
        has_track = len(self.tracks[0]) > 0
        thres = self.new_track_conf_thres[1 if has_track else 0]
        conf = det_cls[unmatched_det_ids].max(dim=-1).values
        indices = (conf > thres).cpu().numpy()
        new_track_det_ids = unmatched_det_ids[indices]
        if not isinstance(new_track_det_ids, np.ndarray): # avoid single element indexing
            new_track_det_ids = [new_track_det_ids]
        # init tracks
        for det_id in unmatched_det_ids[indices]:
            track_idx = self.num_ids[0]
            query = det_app[det_id]
            box = det_box[det_id]
            label = det_cls[det_id].argmax()
            conf = det_cls[det_id][label]
            self.tracks[0][self.num_ids[0]] = Tracklet(track_idx, frame_id, query, box, conf, label)
            self.num_ids[0] += 1
            
        self.terminate_tracks(frame_id)
        self.update_tracks(frame_id)
        
    def get_track_results(self, frame_id, batch=0):
        boxes = []
        ids = []
        labels = []
        for tid, t in self.tracks[batch].items():
            t: Tracklet
            if frame_id in t.frames and frame_id <= t.last_update and t.instance != Tracklet.FAKE_INSTANCE_ID:
                ids.append(tid)
                boxes.append(t.boxes[frame_id] + [t.confidences[frame_id]])
                labels.append(t.label)
        return outs2results(
            bboxes=np.array(boxes), 
            labels=np.array(labels), 
            ids=np.array(ids), 
            num_classes=1
        )