# start point to search module for all registries
default_scope = "mmpretrain"

# checkpoint saving
validation_interval = 1
save_best = 'auto'
checkpoint_config = dict(interval=validation_interval, save_best=save_best)

log_interval = 10
# yapf:disable
log_config = dict(
    interval=log_interval, hooks=[dict(type="LoggerHook"), dict(type="SuperviselyLoggerHook")]
)
# yapf:enable

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]

classification_mode = "one_label"
