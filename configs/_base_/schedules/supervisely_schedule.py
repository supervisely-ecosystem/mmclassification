# https://pytorch.org/docs/1.7.1/optim.html?module-torch.optim
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.8, momentum=0.9, weight_decay=0.0001, nesterov=True)
#optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001)

#optimizer = dict(type='AdamW', lr=0.0015, weight_decay=0.3)
#optimizer_config = dict(grad_clip=dict(max_norm=1.0))

optimizer_config = dict(grad_clip=None)

runner = dict(type='EpochBasedRunner', max_epochs=100)
#log_config = dict(interval=100)



#mmsegmentation +
#runner = dict(type='IterBasedRunner', max_iters=160000)
#checkpoint_config = dict(by_epoch=False, interval=16000)
#evaluation = dict(interval=16000, metric='mIoU')