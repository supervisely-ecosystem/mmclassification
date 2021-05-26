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