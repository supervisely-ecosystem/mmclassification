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


# # settings:
# _base_/models
# num_classes=1000
#
# _base_/datasets
# size=(256, -1) - input size
# resize pillow in VGG???
# dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
# samples_per_gpu=32,
# workers_per_gpu=2
# test=dict( ype=dataset_type, data_prefix='val', pipeline=test_pipeline) - drop test pipeline??
# sly_dataset - load CLASSES field
#
# _base_/schedules
# max_epochs=100
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
# lr_config = dict(policy='step', step=[100, 150])
# lr_config = dict(policy='CosineAnnealing', min_lr=0)
# lr_config = dict(policy='step', gamma=0.98, step=1)

# #runtime
# log interval=100
# hook dict(type='TextLoggerHook')
# load_from = None
# resume_from = None


#@TODO: find configs for models ResNeSt-50 , ResNeSt-101, ResNeSt-200, ResNeSt-269
#@TODO: for custom weights - load from and resume from???
if __name__ == "__main__":
    #sly.main_wrapper("main", main)
    main() # for debug