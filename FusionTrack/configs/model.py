assigner_cfg = dict(
    type='HungarianAssigner',
    cls_cost=dict(type='FocalLossCost', weight=2.0),
    reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
    iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)
)

num_queries = 300
low_conf_thres = 0.2

# matcher_cfg = dict(
#     type='TwoStageMatcher',
#     first_match_conf_thres=0.6,
#     first_match_weights=dict(app=8.0, iou=1.0, reg=1.0),
#     first_match_cost_thres=1.0,
#     second_match_conf_thres=low_conf_thres,
#     second_match_weights=dict(app=1.0, iou=2.0, cls=1.0, reg=2.0, motion=4.0),
#     second_match_cost_thres=1.0,
# )

matcher_cfg = dict(
    type='SingleStageMatcher',
    cost_weights=dict(app=1.0, iou=4.5, reg=4.5),
    cost_thres=2.0,
)

fuser_cfg = dict(
    type="MultiLevelQueryFuser",
    d_query=256, 
    d_ffn=1024, 
    d_out=256, 
    num_layers=2,
    n_heads=8,
    n_intermidiates=7, # 6 decoder level + 1 initial query
    ffn_act=dict(type="GELU"),
    ffn_norm=dict(type="LN"),
    dropout=0.1,
    # -------------------------- for Ablation Study -------------------------- #
    use_score_embed=True,
    only_last_query=False
)

tracker_cfg = dict(
    fuser_cfg=fuser_cfg,
    converter_cfg=dict(
        d_query=256,
        d_ffn=1024,
        d_out=256,
        num_layers=2,
        act=dict(type="GELU"),
        norm=dict(type="LN"),
        dropout=0.1,
    ),
    reid_cfg=dict(
        embed_dim=256,
        feedforward_dim=512,
        output_dim=256,
        num_fcs=2,
        add_identity=True,
    ),
    # ------------------------- matching configs ------------------------- #
    matcher_cfg=matcher_cfg,
    det_preprocess_cfg=dict(
        filter_cfgs=[
            dict(type="NMSFilter", nms_iou_thres=0.5),
            dict(type="ConfidenceFilter", conf_thres=low_conf_thres),
        ]
    ),
    track_preprocess_cfg=dict(motion_iou_thres=0.0, inter_iou_thres=1.0),
    keep_frame=5,
    new_track_conf_thres=[0.5, 0.8],    # used to create new tracklets (first frame / others)
    continue_track_conf_thres=0.8,      # used to continue tracks using unmatched tracks
    # --------------------------- training configs --------------------------- #
    train_cfg=dict(assigner=assigner_cfg),
    app_loss=dict(type='MSELoss', loss_weight=4.0),
    # ---------------------------- extra features ---------------------------- #
    use_score_embed=True,
    use_time_embed=True,
)

model = dict(
    tracker_cfg=tracker_cfg,
    num_queries=num_queries,
    multi_match_k=1,
    freeze_detector=False,
    obj_detect_loss_weight=1.0,
    track_detect_loss_weight=1.0,
)

    
# -------------------------------- interfaces -------------------------------- #
    
import copy
    
def get_model_cfg(expname):
    """used to get model config for different experiments"""
    cfg = copy.deepcopy(model)
    if "no-MLQF" in expname:
        cfg["tracker_cfg"]["fuser_cfg"]["use_score_embed"] = False
        cfg["tracker_cfg"]["fuser_cfg"]["only_last_query"] = True
    if "MLQF-no-lvl" in expname:
        cfg["tracker_cfg"]["fuser_cfg"]["only_last_query"] = True
    if "MLQF-no-score" in expname:
        cfg["tracker_cfg"]["fuser_cfg"]["use_score_embed"] = False
    if "no-QPM" in expname:
        cfg["tracker_cfg"]["use_score_embed"] = False
        cfg["tracker_cfg"]["use_time_embed"] = False
    if "NoMatch" in expname:
        cfg["tracker_cfg"]["det_preprocess_cfg"]["filter_cfgs"].append(dict(type="TrackIoUFilter", iou_thres=0.3))
        cfg["tracker_cfg"]["matcher_cfg"] = dict(type="DoNothingMatcher")
        cfg["tracker_cfg"]["continue_track_conf_thres"] = 0.4
    if "NoMatch2" in expname:
        cfg["tracker_cfg"]["det_preprocess_cfg"]["filter_cfgs"].append(dict(type="TrackIoUFilter", iou_thres=0.2))
        cfg["tracker_cfg"]["matcher_cfg"] = dict(type="DoNothingMatcher")
        cfg["tracker_cfg"]["continue_track_conf_thres"] = 0.3
    if "NoMatch3" in expname:
        cfg["tracker_cfg"]["det_preprocess_cfg"]["filter_cfgs"].append(dict(type="TrackIoUFilter", iou_thres=0.2))
        cfg["tracker_cfg"]["matcher_cfg"] = dict(type="DoNothingMatcher")
        cfg["tracker_cfg"]["continue_track_conf_thres"] = 0.2
    if "NoMatch4" in expname:
        cfg["tracker_cfg"]["det_preprocess_cfg"]["filter_cfgs"].append(dict(type="TrackIoUFilter", iou_thres=0.3))
        cfg["tracker_cfg"]["matcher_cfg"] = dict(type="DoNothingMatcher")
        cfg["tracker_cfg"]["continue_track_conf_thres"] = 0.2
    if "NoMatch5" in expname:
        cfg["tracker_cfg"]["det_preprocess_cfg"]["filter_cfgs"].append(dict(type="TrackIoUFilter", iou_thres=0.4))
        cfg["tracker_cfg"]["matcher_cfg"] = dict(type="DoNothingMatcher")
        cfg["tracker_cfg"]["continue_track_conf_thres"] = 0.1
    if "TrackNMS" in expname:
        cfg["tracker_cfg"]["track_preprocess_cfg"]["inter_iou_thres"] = 0.9
    if "raw2" in expname:
        cfg["tracker_cfg"]["matcher_cfg"]["cost_weights"] = dict(app=1.0, iou=4.5, reg=4.5)
        cfg["tracker_cfg"]["matcher_cfg"]["cost_thres"] = 3.0
    if "raw3" in expname:
        cfg["tracker_cfg"]["matcher_cfg"]["cost_weights"] = dict(app=0.5, iou=5.0, reg=5.0)
        cfg["tracker_cfg"]["matcher_cfg"]["cost_thres"] = 2.0
    if "raw4" in expname:
        cfg["tracker_cfg"]["matcher_cfg"]["cost_weights"] = dict(app=0.5, iou=5.0, reg=5.0)
        cfg["tracker_cfg"]["matcher_cfg"]["cost_thres"] = 1.5
    return cfg