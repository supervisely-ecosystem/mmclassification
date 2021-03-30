# optimizer - learn more in pytorch docs
# https://pytorch.org/docs/1.7.1/optim.html?module-torch.optim

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001)

optimizer_config = dict(grad_clip=None)

# learning policy

lr_config = dict(policy='step', step=[30, 60, 90])
# lr_config = dict(policy='CosineAnnealing', min_lr=0)
# lr_config = dict(policy='step', gamma=0.98, step=1)

runner = dict(type='EpochBasedRunner', max_epochs=100)