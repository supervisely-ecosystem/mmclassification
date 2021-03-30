# dataset settings
dataset_type = 'Supervisely'

batch_size = 32
target_width = 224
target_height = 224
workers_per_gpu = 2

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # list of augmentations
    dict(type='Resize', size=(target_width, target_height)),
    #dict(type='RandomResizedCrop', size=(target_width, target_height)),
    #dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),

    # do not modify this
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(target_width, target_height)),

    # do not modify this
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=workers_per_gpu,
    train=dict(
        type=dataset_type,
        data_prefix='train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='val',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')
