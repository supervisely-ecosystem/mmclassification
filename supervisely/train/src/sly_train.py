import os
import supervisely_lib as sly

import sly_globals as g
import ui as ui
import architectures
from sly_train_progress import get_progress_cb
from sly_train_args import init_script_arguments
from splits import get_train_val_sets, verify_train_val_sets
import train_config


@g.my_app.callback("train")
@sly.timeit
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

        # split to train / validation sets (paths to images and annotations)
        train_set, val_set = get_train_val_sets(project_dir, state)
        verify_train_val_sets(train_set, val_set)
        sly.logger.info(f"Train set: {len(train_set)} images")
        sly.logger.info(f"Val set: {len(val_set)} images")

        # # convert Supervisely project to YOLOv5 format
        # progress_cb = get_progress_cb("Convert Supervisely to YOLOv5 format", len(train_set) + len(val_set))
        # yolov5_format.transform(project_dir, train_data_dir, train_set, val_set, progress_cb)
        #
        # init sys.argv for main training script

        train_config.generate(state)

        init_script_arguments(state, project_dir)
        #Config._validate_py_syntax(filename)
        from tools.train import main as mm_train #@TODO: move to imports section on top
        # _base_ = [
        #     '../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py',
        #     '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
        # ]


        # settings:
        # --gpus
        # --gpu-ids

        # models:
        # cfg.model.head.num_classes

        # datasets:
        # samples_per_gpu=32,
        # workers_per_gpu=2,
        # evaluation = dict(interval=1, metric='accuracy')
        # target_size = 256

        # schedules:
        # optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
        # optimizer_config = dict(grad_clip=None)
        # lr_config = dict(policy='step', step=[30, 60, 90])
        # lr_config = dict(policy='CosineAnnealing', min_lr=0)
        # lr_config = dict(policy='step', gamma=0.98, step=1)
        # runner = dict(type='EpochBasedRunner', max_epochs=100)

        # runtime:
        # hooks = logging hook
        # log interval=100,
        # checkpoint_config = dict(interval=1)
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
        g.my_app.show_modal_window(f"Oops! Something went wrong, please try again or contact tech support. "
                                   f"Find more info in the app logs. Error: {repr(e)}", level="error")
        api.app.set_field(task_id, "state.started", False)
        raise e  #@TODO: uncomment only for debug

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

# @TODO: check if it is used correctly: validation_interval vs (and) logging frequence
#@TODO: custom weights - load-from option

#@TODO: * in model name - что это?
#@TODO: add predicted tags (gt_labels.json) to model file
#@TODO: separate - update content and options in comparegallery
#@TODO: disable preview button if custom pipeline is not defined
#@TODO: preview augentations
#@TODO: random weights initialization?
#@TODO: --resume-from - continue training
if __name__ == "__main__":
    sly.main_wrapper("main", main)
