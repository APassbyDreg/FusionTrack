img_scale = (800, 1440)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

train_vid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_instance=True),
    dict(
        type='RandomAffine',
        max_rotate_degree=5.0,
        max_translate_ratio=0.05,
        scaling_ratio_range=(0.9, 1.2),
        max_shear_degree=2.5,
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
        keep_ratio=True,
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