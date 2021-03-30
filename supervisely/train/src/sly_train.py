import os
import supervisely_lib as sly
import sly_ui as ui
import sly_globals as g


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "modal.state.slyProjectId": g.project_id,
    })

    data = {}
    state = {}
    data["taskId"] = g.task_id

    # read project information and meta (classes + tags)
    g.init_project_info_and_meta()

    # init data for UI widgets
    ui.init(data, state)

    g.app.run(data=data, state=state)

# topk=(1, 5) - configure?

# # settings:
# _base_/models
# num_classes=1000
#
# _base_/datasets
# size=(256, -1) - input size - OK
# random crop - 224 x 224 ??? - OK

# epochs  - OK
# resize pillow in VGG???
# dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'), - OK
# vertical flip - OK


# samples_per_gpu=32, - OK
# workers_per_gpu=2 - OK
# test=dict( ype=dataset_type, data_prefix='val', pipeline=test_pipeline) - drop test pipeline??
# sly_dataset - load CLASSES field
#
# _base_/schedules
# max_epochs=100 - OK
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001) - OK

# grad_clip-????

# lr_config = dict(policy='step', step=[100, 150])
# lr_config = dict(policy='CosineAnnealing', min_lr=0)
# lr_config = dict(policy='step', gamma=0.98, step=1)

# #runtime
# log interval=100
# hook dict(type='TextLoggerHook')
# load_from = None
# resume_from = None

# add to readme - added value + keep main ideology
#@TODO: total tags, selected tags
#@TODO: minimum instance versio - new widgets sly-size
#@TODO: find configs for models ResNeSt-50 , ResNeSt-101, ResNeSt-200, ResNeSt-269
#@TODO: for custom weights - load from and resume from???
if __name__ == "__main__":
    sly.main_wrapper("main", main)
