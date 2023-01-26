# model settings

import os.path as osp
from pcs.augmentations import PCSDefaultAugmentor, PCSMinimalAugmentationList, PCSContrastAugmentationList, PCSPerlinAugmentationList
name = "bns2a"
deep_stem=True
norm_cfg = dict(type='BN', requires_grad=True)
inplanes = 48
in_channels = [(inplanes * 4) * 2**i for i in range(4)]
head_in_channels = in_channels[0]
skip_max_pool=True
stem_stride=True
resolution_reduction = 2 ** (int(not skip_max_pool) + int(stem_stride))
first_anchor = 8
first_anchor_step = 4
anchor_scales = [first_anchor // resolution_reduction]
anchor_strides = [2 * 2**i for i in range(5)]
augmentation_list=PCSMinimalAugmentationList
model = dict(
    type='S2ANetDetector',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        deep_stem=True,
        depth=50,
        skip_max_pool=skip_max_pool,
        stem_kernel_size=3,
        stem_stride=stem_stride,
        in_channels=1,
        inplanes=inplanes,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=norm_cfg,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=in_channels,
        out_channels=head_in_channels,
        add_extra_convs=True,
        num_outs=5,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='S2ANetHead',
        num_classes=2,
        in_channels=head_in_channels,
        feat_channels=head_in_channels,
        stacked_convs=2,
        with_orconv=True,
        anchor_ratios=[1.0],
        anchor_strides=anchor_strides,
        anchor_scales=anchor_scales,
        target_means=[.0, .0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
        loss_fam_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_fam_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_odm_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_odm_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)))

# training and testing settings
train_cfg = dict(
    fam_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='BboxOverlaps2D_rotated')),
        bbox_coder=dict(type='DeltaXYWHABBoxCoder',
                        target_means=(0., 0., 0., 0., 0.),
                        target_stds=(1., 1., 1., 1., 1.),
                        clip_border=True),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    odm_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='BboxOverlaps2D_rotated')),
        bbox_coder=dict(type='DeltaXYWHABBoxCoder',
                        target_means=(0., 0., 0., 0., 0.),
                        target_stds=(1., 1., 1., 1., 1.),
                        clip_border=True),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))


test_cfg = dict(
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms_rotated', iou_thr=0.1),
    max_per_img=2000)

# dataset settings
dataset_type = 'PCSTrainDataset'
data_root = '/dss/dsshome1/lxc08/ga82gan2/datasets/pcs/'

img_norm_cfg = dict(
    mean=[127.5], std=[61.2], to_rgb=False)
train_pipeline = [] # managed by dataset loader
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(384, 384),
        flip=False,
        transforms=[
            dict(type='RotatedResize', img_scale=(384, 384), keep_ratio=True),
            dict(type='RotatedRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

imgs_per_gpu = 2 if norm_cfg["type"] == "GN" else 32
data = dict(
    imgs_per_gpu=imgs_per_gpu,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'pcs_train.pkl',
        img_prefix=data_root + 'pcs_train/',
        pipeline=train_pipeline,
        img_mean=augmentation_list.img_mean,
        img_std=augmentation_list.img_std,
        augmentation_list=augmentation_list),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'pcs_validation.pkl',
        img_prefix=data_root + 'pcs_validation/',
        pipeline=test_pipeline)
)

optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(norm_decay_mult=0)
)

optimizer_config = dict(
    grad_clip=dict(max_norm=5, norm_type=2)
)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=20000,
    warmup_ratio=1.0 / 3,
    step=[800, 900])

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

# runtime settings
total_epochs = 10000
dist_params = dict(backend='nccl')
log_level = 'DEBUG'
latest_path = f"/dss/dsshome1/lxc08/ga82gan2/repos/s2anet/work_dirs/{name}/latest.pth"
if not osp.exists(latest_path):
    latest_path = None
resume_from=latest_path
load_from=None
# resume_from=None
# load_from=latest_path
workflow = [('train', 1)]

