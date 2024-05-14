model = dict(
    type='Recognizer3D',
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
        patch_norm=True),
    cls_head=dict(
        type='I3DHead',
        in_channels=1024,
        num_classes=11,
        spatial_type='avg',
        dropout_ratio=0.5,
        loss_cls=dict(
            type='CrossEntropyLoss')),
    test_cfg=dict(average_clips='prob'))
checkpoint_config = dict(interval=50)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/data/.../VST_fusion_dataset/VST/checkpoints/swin_base_patch244_window877_kinetics600_22k.pth'
resume_from = '/data/.../VST_fusion_dataset/workdir/4ch_cine_11cls/spacing_0.994/TRAIN/epoch_100.pth'
workflow = [('train', 1)]
spacing = 0.994
padding = 210
dataset_type = 'RawframeDataset'
data_root = '/.../VST/data/masked_sax_cine'
data_root_val = '/.../VST/data/masked_sax_cine'
ann_file_train = '/data/.../VST_fusion_dataset/workdir/Diagnosis_ann/4ch_cine/0.994.txt'
ann_file_val = '/data/.../VST_fusion_dataset/workdir/annotations/external/4ch_cine_0.994_11cls_test.txt'
ann_file_test = '/data/.../VST_fusion_dataset/workdir/annotations/external/4ch_cine_0.994_11cls_test.txt'
mask_ann = '/data/.../VST_fusion_dataset/workdir/mask_ann_map.pkl'
train_pipeline = [
    dict(type='SampleFrames', clip_len=13, frame_interval=2, num_clips=1),
    dict(
        type='NIIDecodeV2',
        mask_ann='/data/.../VST_fusion_dataset/workdir/mask_ann_map.pkl'),
    dict(type='Padding', size=(210, 210)),
    dict(type='SingleNorm'),
    dict(type='Imgaug', transforms=[dict(type='Rotate', rotate=(-20, 20))]),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='ColorJitter', color_space_aug=True),
    dict(type='AddRandomNumber', range=(-0.1, 0.1)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=13,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(
        type='NIIDecodeV2',
        mask_ann='/data/.../VST_fusion_dataset/workdir/mask_ann_map.pkl'),
    dict(type='Padding', size=(210, 210)),
    dict(type='SingleNorm'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=13,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(
        type='NIIDecodeV2',
        mask_ann='/data/.../VST_fusion_dataset/workdir/mask_ann_map.pkl'),
    dict(type='Padding', size=(210, 210)),
    dict(type='SingleNorm'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    type='ClassBalancedDataset',
    oversample_thr=0.2,
    val_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=5, workers_per_gpu=4),
    train=dict(
        type='RawframeDataset',
        ann_file=
        '/data/.../VST_fusion_dataset/workdir/Diagnosis_ann/4ch_cine/0.994.txt',
        data_prefix='/.../VST/data/masked_sax_cine',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=13,
                frame_interval=2,
                num_clips=1),
            dict(
                type='NIIDecodeV2',
                mask_ann=
                '/data/.../VST_fusion_dataset/workdir/mask_ann_map.pkl'),
            dict(type='Padding', size=(210, 210)),
            dict(type='SingleNorm'),
            dict(
                type='Imgaug',
                transforms=[dict(type='Rotate', rotate=(-20, 20))]),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='ColorJitter', color_space_aug=True),
            dict(type='AddRandomNumber', range=(-0.1, 0.1)),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        num_classes=11,
        fusion=False),
    val=dict(
        type='RawframeDataset',
        ann_file=
        '/data/.../VST_fusion_dataset/workdir/annotations/external/4ch_cine_0.994_11cls_test.txt',
        data_prefix='/.../VST/data/masked_sax_cine',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=13,
                frame_interval=2,
                num_clips=1,
                test_mode=True),
            dict(
                type='NIIDecodeV2',
                mask_ann=
                '/data/.../VST_fusion_dataset/workdir/mask_ann_map.pkl'),
            dict(type='Padding', size=(210, 210)),
            dict(type='SingleNorm'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        num_classes=11,
        fusion=False),
    test=dict(
        type='RawframeDataset',
        ann_file=
        '/data/.../VST_fusion_dataset/workdir/annotations/external/4ch_cine_0.994_11cls_test.txt',
        data_prefix='/.../VST/data/masked_sax_cine',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=13,
                frame_interval=2,
                num_clips=1,
                test_mode=True),
            dict(
                type='NIIDecodeV2',
                mask_ann=
                '/data/.../VST_fusion_dataset/workdir/mask_ann_map.pkl'),
            dict(type='Padding', size=(210, 210)),
            dict(type='SingleNorm'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        num_classes=11,
        fusion=False))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])
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
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5)
total_epochs = 300
work_dir = '/data/.../VST_fusion_dataset/workdir/4ch_cine_11cls/spacing_0.994/TRAIN'
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
