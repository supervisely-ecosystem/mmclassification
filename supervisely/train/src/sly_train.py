import os
import supervisely_lib as sly
import sly_globals as g
import ui as ui
from sly_train_progress import get_progress_cb
from sly_train_args import init_script_arguments


@g.my_app.callback("train")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    try:
        # # convert Supervisely project to YOLOv5 format
        # progress_cb = get_progress_cb("Convert Supervisely to YOLOv5 format", len(train_set) + len(val_set))
        # yolov5_format.transform(project_dir, train_data_dir, train_set, val_set, progress_cb)

        # init sys.argv for main training script
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

#@TODO: debug error imgcorruptlike
#@TODO: scroll to active step - after restart and on finish
#@TODO: tags cooccurance - добавить в readme, поменять там табличный виджет, сделать зафиксированные колонки и скрол горизонтальный
#@TODO: убрать маргины с карточек
#@TODO: save_set_to_json - save in imagenet format, rename clean_bad_images - add filed - tag index and save to json for our custom dataset
#@TODO: save session link in artifacts dir
#@TODO: state["workersPerGPU"] = 0# 2  # 0 - for debug @TODO: for debug
#@TODO: validate project size after project cleaning
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
#@TODO: readme - tags co-occurrence-matrix
#Oops! Something went wrong, please try again or contact tech support. Find more info in the app logs. Error: AttributeError("'dict' object has no attribute 'optimizer'")
# уменьшить скругления + сделать кликабельность по названию?
#@TODO: add to readme - unpack KV tag
#@TODO: min version instance

# implement save_best renaming
if __name__ == "__main__":
    sly.main_wrapper("main", main)
