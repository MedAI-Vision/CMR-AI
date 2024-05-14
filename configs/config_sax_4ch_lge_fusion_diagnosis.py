# model architecture
model = dict(
    type='fusion_model',
    backbone=dict(
        type='SwinTransformer3D',
        patch_size=(2, 4, 4),
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=(8, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        frozen_stages=-1,
        patch_norm=True),
    cls_head=dict(
        type='fusion_ConcatHead',
        in_channels=1024,
        num_classes=11,
        spatial_type='avg',
        dropout_ratio=0.5,
        num_mod=3,
        loss_cls=dict(
            type='CrossEntropyLoss')),
    test_cfg=dict(average_clips='prob'),
    fusion=True,
    sax_weight=
    '/data/.../VST_fusion_dataset/workdir/sax_cine_11cls/spacing_0.994/TRAIN-12-3/epoch_300_fusion_base.pth',
    ch_weight=
    '/data/.../VST_fusion_dataset/workdir/4ch_cine_11cls/spacing_0.994/TRAIN-12-3/epoch_300_fusion_base.pth',
    lge_weight=
    '/data/.../VST_fusion_dataset/workdir/sax_lge_11cls/spacing_0.994/TRAIN12-3-noseed/epoch_300_fusion_base.pth'
)

# some basic experiment params and required data and annotation path
checkpoint_config = dict(interval=5)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
train_fold = 12
test_fold = 3
mask_ann = '/data/.../VST_fusion_dataset/workdir/mask_ann_map.pkl'
spacing = 0.994
padding = 210
dataset_type = 'RawframeDataset'
data_root = '/.../VST/data/masked_sax_cine'
data_root_val = '/.../VST/data/masked_sax_cine'
ann_file_train = '/data/.../VST_fusion_dataset/workdir/Diagnosis_ann/lge_fusion/0.994_fold_12.txt'
ann_file_val = '/data/.../VST_fusion_dataset/workdir/Diagnosis_ann/lge_fusion/0.994_fold_3.txt'
ann_file_test = '/data/.../VST_fusion_dataset/workdir/annotations/external/sax_4ch_lge_0.994_11cls_fusion_test_exclude2.txt'
img_norm_cfg = dict(
    mean=[68.95, 71.7, 121.02], std=[56.96, 54.65, 39.4], to_bgr=False)

# train process params, included data augmentation selection
train_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=13,
        frame_interval=2,
        num_clips=1,
        lge_clip_len=9,
        lge_frame_interval=1,
        lge_num_clips=1),
    dict(
        type='NIIDecodeV2',
        mask_ann='/data/.../VST_fusion_dataset/workdir/mask_ann_map.pkl'),
    dict(type='Padding', size=(210, 210)),
    dict(type='SingleNorm'),
    dict(type='Imgaug', transforms=[dict(type='Rotate', rotate=(-45, 45))]),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='ColorJitter', color_space_aug=True),
    dict(type='Flip_Z'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['sax', 'ch', 'lge', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['sax', 'ch', 'lge', 'label'])
]

# validation params
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=13,
        frame_interval=2,
        num_clips=1,
        lge_clip_len=9,
        lge_frame_interval=1,
        lge_num_clips=1,
        test_mode=True),
    dict(
        type='NIIDecodeV2',
        mask_ann='/data/.../VST_fusion_dataset/workdir/mask_ann_map.pkl'),
    dict(type='Padding', size=(210, 210)),
    dict(type='SingleNorm'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['sax', 'ch', 'lge', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['sax', 'ch', 'lge', 'label'])
]

# testing params
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=13,
        frame_interval=2,
        num_clips=1,
        lge_clip_len=9,
        lge_frame_interval=1,
        lge_num_clips=1,
        test_mode=True),
    dict(
        type='NIIDecodeV2',
        mask_ann='/data/.../VST_fusion_dataset/workdir/mask_ann_map.pkl'),
    dict(type='Padding', size=(210, 210)),
    dict(type='SingleNorm'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['sax', 'ch', 'lge', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['sax', 'ch', 'lge', 'label'])
]

# dataset and dataloader params
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    type='ClassBalancedDataset',
    oversample_thr=0.2,
    val_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1, workers_per_gpu=4),
    train=dict(
        type='RawframeDataset',
        ann_file=
        '/data/.../VST_fusion_dataset/workdir/Diagnosis_ann/lge_fusion/0.994_fold_12.txt',
        data_prefix='/.../VST/data/masked_sax_cine',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=13,
                frame_interval=2,
                num_clips=1,
                lge_clip_len=9,
                lge_frame_interval=1,
                lge_num_clips=1),
            dict(
                type='NIIDecodeV2',
                mask_ann=
                '/data/.../VST_fusion_dataset/workdir/mask_ann_map.pkl'),
            dict(type='Padding', size=(210, 210)),
            dict(type='SingleNorm'),
            dict(
                type='Imgaug',
                transforms=[dict(type='Rotate', rotate=(-45, 45))]),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='ColorJitter', color_space_aug=True),
            dict(type='Flip_Z'),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(
                type='Collect',
                keys=['sax', 'ch', 'lge', 'label'],
                meta_keys=[]),
            dict(type='ToTensor', keys=['sax', 'ch', 'lge', 'label'])
        ],
        num_classes=11,
        fusion=True,
        type1='sax',
        type2='4ch',
        type3='lge'),
    val=dict(
        type='RawframeDataset',
        ann_file=
        '/data/.../VST_fusion_dataset/workdir/Diagnosis_ann/lge_fusion/0.994_fold_3.txt',
        data_prefix='/.../VST/data/masked_sax_cine',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=13,
                frame_interval=2,
                num_clips=1,
                lge_clip_len=9,
                lge_frame_interval=1,
                lge_num_clips=1,
                test_mode=True),
            dict(
                type='NIIDecodeV2',
                mask_ann=
                '/data/.../VST_fusion_dataset/workdir/mask_ann_map.pkl'),
            dict(type='Padding', size=(210, 210)),
            dict(type='SingleNorm'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(
                type='Collect',
                keys=['sax', 'ch', 'lge', 'label'],
                meta_keys=[]),
            dict(type='ToTensor', keys=['sax', 'ch', 'lge', 'label'])
        ],
        num_classes=11,
        fusion=True,
        type1='sax',
        type2='4ch',
        type3='lge'),
    test=dict(
        type='RawframeDataset',
        ann_file=
        '/data/.../VST_fusion_dataset/workdir/annotations/external/sax_4ch_lge_0.994_11cls_fusion_test_exclude2.txt',
        data_prefix='/.../VST/data/masked_sax_cine',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=13,
                frame_interval=2,
                num_clips=1,
                lge_clip_len=9,
                lge_frame_interval=1,
                lge_num_clips=1,
                test_mode=True),
            dict(
                type='NIIDecodeV2',
                mask_ann=
                '/data/.../VST_fusion_dataset/workdir/mask_ann_map.pkl'),
            dict(type='Padding', size=(210, 210)),
            dict(type='SingleNorm'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(
                type='Collect',
                keys=['sax', 'ch', 'lge', 'label'],
                meta_keys=[]),
            dict(type='ToTensor', keys=['sax', 'ch', 'lge', 'label'])
        ],
        num_classes=11,
        fusion=True,
        type1='sax',
        type2='4ch',
        type3='lge'))

# evaluation params when in training
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer choice
optimizer = dict(
    type='AdamW',
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.1))))

# learning rate
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5)
total_epochs = 20
work_dir = '/data/.../VST_fusion_dataset/workdir/sax_4ch_lge_fusion_11cls/spacing_0.994/TRAIN-12-3'
find_unused_parameters = False
fp16 = None
optimizer_config = dict(
    type='DistOptimizerHook',
    update_interval=8,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True)
gpu_ids = range(0, 8)
omnisource = False
module_hooks = []
