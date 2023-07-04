# ------------------------------ dataset related ----------------------------- #
def make_clip_sampler(clip_length, avg_frame_interval=2):
    return dict(    
        frame_range=avg_frame_interval * clip_length,
        stride=(clip_length + 1) // 2,
        num_ref_imgs=clip_length,
        filter_key_img=True,
        method='uniform',
        return_key_img=False
    )

# ------------------------ detector pretrain datasets ------------------------ #

crowdhuman_pretrain = dict(
    type="CocoDataset",
    ann_file=f"data/CrowdHuman/annotations/train_cocoformat.json",
    img_prefix=f"data/CrowdHuman/Images",
    classes=("person", ),
    pipeline=[],
)

# ------------------------------ normal datasets ----------------------------- #

mot_train = {
    year: dict(
        type="MOTChallengeDataset",
        detection_file=f"data/MOT/MOT{year}/annotations/train_detections.pkl",
        ann_file=f"data/MOT/MOT{year}/annotations/train_cocoformat.json",
        img_prefix=f"data/MOT/MOT{year}/train",
        classes=("pedestrian", ),
        pipeline=[],
        key_img_sampler=dict(interval=1),
        ref_img_sampler=None,
        filter_empty_gt=False
    )
    for year in [15, 16, 17, 20]
}

dance_track = {
    mode: dict(
        type="DanceTrackDataset",
        ann_file=f"data/DanceTrack/annotations/{mode}_cocoformat.json",
        img_prefix=f"data/DanceTrack/{mode}",
        classes=("pedestrian", ),
        pipeline=[],
        key_img_sampler=dict(interval=1),
        ref_img_sampler=None,
        filter_empty_gt=False
    ) for mode in ['train', 'val', 'test']
}

# -------------------------------- preprocess -------------------------------- #
img_scale = (1440, 800)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

train_img_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_instance=True),
    dict(
        type='RandomAffine',
        max_rotate_degree=5.0,
        max_translate_ratio=0.1,
        scaling_ratio_range=(0.7, 1.5),
        max_shear_degree=2,
        border=(0, 0),
        border_val=(114, 114, 114),
        min_bbox_size=2,
        min_area_ratio=0.2,
        max_aspect_ratio=20,
        bbox_clip_border=True,
    ),
    dict(
        type='Resize',
        img_scale=img_scale,
        keep_ratio=False,
        bbox_clip_border=False
    ),
    dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels', 'gt_instance_ids']),
    dict(
        type='Collect', 
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'img_info')
    )
]

train_vid_pipeline = []

inference_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='VideoCollect', keys=['img'])
]

# ----------------------------------- test ----------------------------------- #

mot_inference = {
    year: dict(
        type="MOTChallengeDataset",
        detection_file=f"data/MOT/MOT{year}/annotations/half-train_detections.pkl",
        ann_file=f"data/MOT/MOT{year}/annotations/half-train_cocoformat.json",
        img_prefix=f"data/MOT/MOT{year}/train",
        pipeline=inference_pipeline,
        ref_img_sampler=None,
    )
    for year in [15, 16, 17, 20]
}

dancetrack_inference = {
    mode: dict(
        type='DanceTrackDataset',
        ann_file=f"data/DanceTrack/annotations/{mode}_cocoformat.json",
        img_prefix=f"data/DanceTrack/{mode}",
        ref_img_sampler=None,
        pipeline=inference_pipeline,
        test_load_ann=(mode=='test'),
        test_mode=(mode=='test'),
        
    ) 
    for mode in ['val', 'test']
}

mot_inference = {
    mode: {
        year: dict(
            type="MOTChallengeDataset",
            detection_file=f"data/MOT/MOT{year}/annotations/{mode}_detections.pkl",
            ann_file=f"data/MOT/MOT{year}/annotations/{mode}_cocoformat.json",
            img_prefix=f"data/MOT/MOT{year}/train" if "val" in mode else f"data/MOT/MOT{year}/{mode}",
            ref_img_sampler=None,
            pipeline=inference_pipeline,
            test_load_ann=True,
            test_mode=True,
        )
        for year in [15, 16, 17, 20]
    }
    for mode in ["test", "train", "half-val"]
}