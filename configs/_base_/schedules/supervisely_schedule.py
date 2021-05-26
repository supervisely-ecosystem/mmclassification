# https://pytorch.org/docs/1.7.1/optim.html?module-torch.optim
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.8, momentum=0.9, weight_decay=0.0001, nesterov=True)
#optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001)

#optimizer = dict(type='AdamW', lr=0.0015, weight_decay=0.3)
#optimizer_config = dict(grad_clip=dict(max_norm=1.0))
# lr_config = dict(
#     policy='CosineAnnealing',
#     min_lr=0,
#     warmup='linear',
#     warmup_iters=10000,
#     warmup_ratio=1e-4)

optimizer_config = dict(grad_clip=None)

# learning policy:
lr_config = dict(policy='step', step=[30, 60, 90])
# lr_config = dict(policy='CosineAnnealing', min_lr=0)
# lr_config = dict(policy='step', gamma=0.98, step=1) # epoch step
# lr_config = dict(policy='step', step=[60, 120, 160], gamma=0.2)
#@TODO: add support
# optimizer = dict(
#     type='SGD',
#     lr=0.5,
#     momentum=0.9,
#     weight_decay=0.00004,
#     paramwise_cfg=dict(norm_decay_mult=0))
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(
#     policy='poly',
#     min_lr=0,
#     by_epoch=False,
#     warmup='constant',
#     warmup_iters=5000,
# )


# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=2500,
#     warmup_ratio=0.25,
#     step=[30, 60, 90])

runner = dict(type='EpochBasedRunner', max_epochs=100)
#log_config = dict(interval=100)


lr_config = dict(policy='step', step=[100, 150])
lr_config = dict(policy='step', step=[30, 60, 90])
lr_config = dict(policy='step', step=[40, 80, 120])
lr_config = dict(policy='CosineAnnealing', min_lr=0)
lr_config = dict(policy='step', gamma=0.98, step=1)
lr_config = dict(
    policy='poly',
    min_lr=0,
    by_epoch=False,
    warmup='constant',
    warmup_iters=5000,
)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2500,
    warmup_ratio=0.25,
    step=[30, 60, 90])

# imagenet for ResNet-50
lr_config = dict(policy='CosineAnnealing', min_lr=0, warmup='linear', warmup_iters=10000, warmup_ratio=1e-4)

lr_config = dict(policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=2500,
    warmup_ratio=0.25)

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=10000,
    warmup_ratio=1e-4)

#https://github.com/open-mmlab/mmcv/blob/13888df2aa22a8a8c604a1d1e6ac1e4be12f2798/mmcv/runner/hooks/lr_updater.py
#https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/hooks/lr_updater.html


# mmdetection
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 19])

# mmsegmentation
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

#+
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16000, metric='mIoU')