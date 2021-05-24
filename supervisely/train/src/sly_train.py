import os
import supervisely_lib as sly

import sly_globals as g
import ui as ui
from architectures import prepare_weights
from sly_train_progress import get_progress_cb
from splits import get_train_val_sets, verify_train_val_sets


@g.my_app.callback("train")
@sly.timeit
def train(api: sly.Api, task_id, context, state, app_logger):
    try:
        #@TODO: uncomment
        #prepare_weights(state)

        # prepare directory for original Supervisely project
        project_dir = os.path.join(g.my_app.data_dir, "sly_project")

        # if-else only to speedup debug, has no effect in prod
        if sly.fs.dir_exists(project_dir):
            pass
        else:
            sly.fs.mkdir(project_dir, remove_content_if_exists=False)  # clean content for debug, has no effect in prod
            # download and preprocess Sypervisely project (using cache)
            download_progress = get_progress_cb("Download data (using cache)", g.project_info.items_count * 2)
            sly.download_project(api, g.project_id, project_dir, cache=g.my_app.cache, progress_cb=download_progress)

        # split to train / validation sets (paths to images and annotations)
        train_set, val_set = get_train_val_sets(project_dir, state)
        verify_train_val_sets(train_set, val_set)
        sly.logger.info(f"Train set: {len(train_set)} images")
        sly.logger.info(f"Val set: {len(val_set)} images")


        # preprocessing: transform labels to bboxes, filter classes, ...

        #sly.Project.to_detection_task(project_dir, inplace=True)
        #train_classes = state["selectedClasses"]
        #sly.Project.remove_classes_except(project_dir, classes_to_keep=train_classes, inplace=True)
        #if state["unlabeledImages"] == "ignore":
        #    sly.Project.remove_items_without_objects(project_dir, inplace=True)

        # # split to train / validation sets (paths to images and annotations)
        # train_set, val_set = get_train_val_sets(project_dir, state)
        # verify_train_val_sets(train_set, val_set)
        # sly.logger.info(f"Train set: {len(train_set)} images")
        # sly.logger.info(f"Val set: {len(val_set)} images")
        #
        # # prepare directory for data in YOLOv5 format (nn will use it for training)
        # train_data_dir = os.path.join(my_app.data_dir, "train_data")
        # sly.fs.mkdir(train_data_dir, remove_content_if_exists=True)  # clean content for debug, has no effect in prod
        #
        # # convert Supervisely project to YOLOv5 format
        # progress_cb = get_progress_cb("Convert Supervisely to YOLOv5 format", len(train_set) + len(val_set))
        # yolov5_format.transform(project_dir, train_data_dir, train_set, val_set, progress_cb)
        #
        # # init sys.argv for main training script
        # init_script_arguments(state, train_data_dir, g.project_info.name)
        #
        # # start train script
        # api.app.set_field(task_id, "state.activeNames", ["labels", "train", "pred", "metrics"])  # "logs",
        # get_progress_cb("YOLOv5: Scanning data ", 1)(1)
        # train_yolov5.main()
        #
        # # upload artifacts directory to Team Files
        # upload_artifacts(g.local_artifacts_dir, g.remote_artifacts_dir)
        # set_task_output()
    except Exception as e:
        g.my_app.show_modal_window(f"Oops! Something went wrong, please try again or contact tech support. "
                                   f"Find more info in the app logs. Error: {repr(e)}", level="error")
        api.app.set_field(task_id, "state.started", False)

    # stop application
    get_progress_cb("Finished, app is stopped automatically", 1)(1)
    g.my_app.stop()


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "modal.state.slyProjectId": g.project_id,
    })

    g.my_app.compile_template(g.root_source_dir)

    data = {}
    state = {}
    data["taskId"] = g.task_id
    ui.init(data, state)  # init data for UI widgets

    g.my_app.run(data=data, state=state)

#@TODO: custom weights - load-from option
#@ __todo_ base models - '../_base_/models/resnet50_supe.py'
# num_classes=XXX
# topk=(1, 5), - option
# backbone=dict(type='ResNet_CIFAR'
#
#mmcls - sly_dataset
# samples_per_gpu=32,
#     workers_per_gpu=2,
# evaluation = dict(interval=1, metric='accuracy')

#@TODO: add predicted tags to model file
#@TODO: separate - update content and options in comparegallery
#@TODO: disable preview button if custom pipeline is not defined
#@TODO: augs templates
#@TODO: preview augentations
#@TODO: random weights initialization?
#@TODO: --resume-from - continue training
if __name__ == "__main__":
    sly.main_wrapper("main", main)
