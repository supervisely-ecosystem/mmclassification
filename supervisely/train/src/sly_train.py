import os

import input_project
import supervisely_lib as sly

import sly_globals as g
import ui as ui
import architectures

from input_project import clean_bad_images_from_project
from sly_train_progress import get_progress_cb
from sly_train_args import init_script_arguments
from splits import get_train_val_sets, verify_train_val_sets, save_set_to_json
import train_config


@g.my_app.callback("train")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    try:
        architectures.prepare_weights(state)

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

        # save selectedTags -> ground-truth labels
        tag_names = state["selectedTags"]
        gt_labels = {tag_name: idx for idx, tag_name in enumerate(tag_names)}
        sly.json.dump_json_file(gt_labels, os.path.join(project_dir, "gt_labels.json"))

        clean_bad_images_from_project(project_dir)

        # split to train / validation sets (paths to images and annotations)
        try:
            train_set, val_set = get_train_val_sets(project_dir, state)
        except  ValueError as e:
            if "total_count != train_count + val_count" in repr(e):
                pass
            else:
                raise e

        verify_train_val_sets(train_set, val_set)
        save_set_to_json(os.path.join(project_dir, "train_set.json"), train_set)
        save_set_to_json(os.path.join(project_dir, "val_set.json"), val_set)
        sly.logger.info(f"Train set: {len(train_set)} images")
        sly.logger.info(f"Val set: {len(val_set)} images")

        input_project.validate_target_tags()

        # # convert Supervisely project to YOLOv5 format
        # progress_cb = get_progress_cb("Convert Supervisely to YOLOv5 format", len(train_set) + len(val_set))
        # yolov5_format.transform(project_dir, train_data_dir, train_set, val_set, progress_cb)
        # init sys.argv for main training script

        train_config.save_from_state(state)

        init_script_arguments(state)
        from tools.train import main as mm_train #@TODO: move to imports section on top
        mm_train()

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
        api.app.set_field(task_id, "state.started", False)
        raise e  # app will handle this error and show modal window

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
    #sly.fs.clean_dir(g.my_app.data_dir)
    g.my_app.run(data=data, state=state)


# implement save_best renaming
#@TODO: if several training tags are assigned to an image
#@TODO: add ON/OFF for custom augmentations
#@TODO: custom augs - reimplement prepare data in BaseDataset
#@TODO: SuperviselyLoggerHook in default_runtime
#@TODO: move mm_train import on top
#@TODO: runtime load_from / or args --resume-from
#@TODO: release new version of SDK before release app
#@TODO: if OOM error, make a special message for that
#@TODO: custom weights - load-from option
#@TODO: random weights initialization?
#@TODO: --resume-from - continue training
#@TODO: readme - add py-configs to training artifacts
if __name__ == "__main__":
    sly.main_wrapper("main", main)
