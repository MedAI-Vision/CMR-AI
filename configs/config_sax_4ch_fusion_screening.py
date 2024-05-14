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
        patch_norm=True),
    cls_head=dict(
        type='fusion_ConcatHead',
        in_channels=1024,
        num_classes=2,
        spatial_type='avg',
        dropout_ratio=0.5,
        num_mod=2),
    test_cfg=dict(average_clips='prob'),
    fusion=True,
    sax_weight=
    '/data/.../VST_fusion_dataset/workdir/sax_cine_bin/spacing_0.994/TRAIN/epoch_300_fusion_base.pth',
    ch_weight=
    '/data/.../VST_fusion_dataset/workdir/4ch_cine_bin/spacing_0.994/TRAIN/epoch_300_fusion_base.pth'
)
checkpoint_config = dict(interval=10)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
dataset_type = 'RawframeDataset'
spacing = 0.994
padding = 210
data_root = '/.../VST/data/masked_sax_cine'
data_root_val = '/.../VST/data/masked_sax_cine'
ann_file_train = '/data/.../VST_fusion_dataset/workdir/Screen_ann/cine_fusion/0.994.txt'
ann_file_val = '/data/.../VST_fusion_dataset/workdir/Screen_ann/cine_fusion/0.994.txt'
ann_file_test = '/data/.../VST_fusion_dataset/workdir/annotations/external/sax_4ch_cine_0.994_bin_fusion_test.txt'
mask_ann = '/data/.../VST_fusion_dataset/workdir/mask_ann_map.pkl'
img_norm_cfg = dict(mean=[68.95, 71.7, 0], std=[56.96, 54.65, 1], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=13, frame_interval=2, num_clips=1),
    dict(
        type='NIIDecodeV2',
        mask_ann='/data/.../VST_fusion_dataset/workdir/mask_ann_map.pkl'),
    dict(type='Padding', size=(210, 210)),
    dict(type='SingleNorm'),
    dict(type='Imgaug', transforms=[dict(type='Rotate', rotate=(-45, 45))]),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='ColorJitter', color_space_aug=True),
    dict(type='AddRandomNumber', range=(-0.1, 0.1)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['sax', 'ch', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['sax', 'ch', 'label'])
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
    dict(type='Collect', keys=['sax', 'ch', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['sax', 'ch', 'label'])
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
    dict(type='Collect', keys=['sax', 'ch', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['sax', 'ch', 'label'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    type='ClassBalancedDataset',
    oversample_thr=0.85,
    val_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1, workers_per_gpu=4),
    train=dict(
        type='RawframeDataset',
        ann_file=
        '/data/.../VST_fusion_dataset/workdir/Screen_ann/cine_fusion/0.994.txt',
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
                transforms=[dict(type='Rotate', rotate=(-45, 45))]),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='ColorJitter', color_space_aug=True),
            dict(type='AddRandomNumber', range=(-0.1, 0.1)),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['sax', 'ch', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['sax', 'ch', 'label'])
        ],
        num_classes=2,
        fusion=True,
        type1='sax',
        type2='4ch'),
    val=dict(
        type='RawframeDataset',
        ann_file=
        '/data/.../VST_fusion_dataset/workdir/annotations/external/sax_4ch_cine_0.994_bin_fusion_test.txt',
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
            dict(type='Collect', keys=['sax', 'ch', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['sax', 'ch', 'label'])
        ],
        num_classes=2,
        fusion=True,
        type1='sax',
        type2='4ch'),
    test=dict(
        type='RawframeDataset',
        ann_file=
        '/data/.../VST_fusion_dataset/workdir/annotations/external/sax_4ch_cine_0.994_bin_fusion_test.txt',
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
            dict(type='Collect', keys=['sax', 'ch', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['sax', 'ch', 'label'])
        ],
        num_classes=2,
        fusion=True,
        type1='sax',
        type2='4ch'))
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
total_epochs = 20
work_dir = '/data/.../VST_fusion_dataset/workdir/sax_4ch_cine_fusion_bin/spacing_0.994/TRAIN'
find_unused_parameters = False
fp16 = None
optimizer_config = dict(
    type='DistOptimizerHook',
    update_interval=8,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True)
gpu_ids = range(0, 4)
omnisource = False
module_hooks = []
