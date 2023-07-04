import copy
import torch.nn as nn

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.layers import ShapeSpec

from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from detrex.layers import PositionEmbeddingSine

from .transmotion_detector import TransMotionDetectorV3
from .transmotion_transformer import TransMotionTransformer, TransMotionTransformerDecoder, TransMotionTransformerEncoder
from ..transmotion_detector_criterion import TransMotionDetectorTrackCriterion, TransMotionDetectorDetectCriterion

def get_detector(pretrained=False):
    num_det_queries = 300
    
    return TransMotionDetectorV3(
        backbone=ResNet(
            stem=BasicStem(in_channels=3, out_channels=64, norm="FrozenBN"),
            stages=ResNet.make_default_stages(
                depth=50,
                stride_in_1x1=False,
                norm="FrozenBN",
            ),
            out_features=["res3", "res4", "res5"],
            freeze_at=1,
        ),
        position_embedding=PositionEmbeddingSine(
            num_pos_feats=128,
            temperature=10000,
            normalize=True,
            offset=-0.5,
        ),
        neck=ChannelMapper(
            input_shapes={
                "res3": ShapeSpec(channels=512),
                "res4": ShapeSpec(channels=1024),
                "res5": ShapeSpec(channels=2048),
            },
            in_features=["res3", "res4", "res5"],
            out_channels=256,
            num_outs=4,
            kernel_size=1,
            norm_layer=nn.GroupNorm(num_groups=32, num_channels=256),
        ),
        transformer=TransMotionTransformer(
            encoder=TransMotionTransformerEncoder(
                embed_dim=256,
                num_heads=8,
                feedforward_dim=2048,
                attn_dropout=0.0,
                ffn_dropout=0.0,
                num_layers=6,
                post_norm=False,
                num_feature_levels=4,
            ),
            decoder=TransMotionTransformerDecoder(
                embed_dim=256,
                num_heads=8,
                feedforward_dim=2048,
                attn_dropout=0.0,
                ffn_dropout=0.0,
                num_layers=6,
                return_intermediate=True,
                num_feature_levels=4,
                num_det_queries=num_det_queries,
                mask_track2det=True, # TODO: change this as configurable
            ),
            num_feature_levels=4,
            combine_det_track=True
        ),
        embed_dim=256,
        num_classes=1,
        num_queries=num_det_queries,
        aux_loss=True,
        criterion=TransMotionDetectorDetectCriterion(
            num_classes=1,
            matcher=HungarianMatcher(
                cost_class=2.0,
                cost_bbox=5.0,
                cost_giou=2.0,
                cost_class_type="focal_loss_cost",
                alpha=0.25,
                gamma=2.0,
            ),
            weight_dict={
                "loss_class": 1,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
                "loss_class_enc": 1.0,
                "loss_bbox_enc": 5.0,
                "loss_giou_enc": 2.0,
            },
            loss_class_type="focal_loss",
            alpha=0.25,
            gamma=2.0,
        ),
        criterion_track=TransMotionDetectorTrackCriterion(
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            ),
            loss_box=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0)
        ),
        select_box_nums_for_evaluation=num_det_queries,
        seperate_det_track_head=True,
        init_cfg=dict(
            type="Pretrained",
            src="pretrained/detrex_dab_deformable_detr_singlestage_person_only.pth"
        ) if pretrained else {}
    )