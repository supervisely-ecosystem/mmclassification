import os
import supervisely_lib as sly
import sly_globals as g
import ui as ui
from sly_train_progress import _update_progress_ui
from sly_train_args import init_script_arguments
import sly_logger_hook
from functools import partial


def upload_artifacts_and_log_progress():
    fields = [
        {"field": "data.progressEpoch", "payload": None},
        {"field": "data.progressIter", "payload": None},
        {"field": "data.eta", "payload": None},
    ]
    g.api.app.set_fields(g.task_id, fields)

    def upload_monitor(monitor, api: sly.Api, task_id, progress: sly.Progress):
        if progress.total == 0:
            progress.set(monitor.bytes_read, monitor.len, report=False)
        else:
            progress.set_current_value(monitor.bytes_read, report=False)
        _update_progress_ui("UploadDir", g.api, g.task_id, progress)

    progress = sly.Progress("Upload artifacts directory", 0, is_size=True)
    progress_cb = partial(upload_monitor, api=g.api, task_id=g.task_id, progress=progress)

    remote_dir = f"/mmclassification/{g.task_id}_{g.project_info.name}"
    g.api.file.upload_directory(g.team_id, g.artifacts_dir, remote_dir, progress_size_cb=progress_cb)


@g.my_app.callback("train")
@sly.timeit
#@g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    #try:
        # init sys.argv for main training script

    # hide progress bars and eta

    upload_artifacts_and_log_progress()
    return
    init_script_arguments(state)
    from tools.train import main as mm_train #@TODO: move to imports section on top
    mm_train()
    api.file.upload()
    # upload artifacts directory to Team Files
    #upload_artifacts(g.local_artifacts_dir, g.remote_artifacts_dir)
    #set_task_output()
    # except Exception as e:
    #     api.app.set_field(task_id, "state.started", False)
    #     raise e  # app will handle this error and show modal window

    # stop application
    #get_progress_cb("Finished, app is stopped automatically", 1)(1)
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

#@TODO: cooccurance table
# {'mode': 'train', 'epoch': 1, 'iter': 10, 'lr': 0.001, 'memory': 1839, 'data_time': 0.2636455535888672, 'loss': 4.4233519554138185, 'time': 0.35094327926635743}
# Error: AttributeError("'ConfigDict' object has no attribute 'lr_config'")
#@TODO: debug error imgcorruptlike
#@TODO: scroll to active step - after restart and on finish
#@TODO: tags cooccurance - добавить в readme, поменять там табличный виджет, сделать зафиксированные колонки и скрол горизонтальный
#@TODO: save session link in artifacts dir
#@TODO: state["workersPerGPU"] = 0# 2  # 0 - for debug @TODO: for debug
#@TODO: add ON/OFF for custom augmentations
#@TODO: custom augs - reimplement prepare data in BaseDataset
#@TODO: SuperviselyLoggerHook in default_runtime
#@TODO: move mm_train import on top
#@TODO: release new version of SDK before release app
#@TODO: if OOM error, make a special message for that
#@TODO: custom weights - load-from option
#@TODO: readme - add py-configs to training artifacts
#@TODO: readme - tags co-occurrence-matrix
#Oops! Something went wrong, please try again or contact tech support. Find more info in the app logs. Error: AttributeError("'dict' object has no attribute 'optimizer'")
#@TODO: add to readme - unpack KV tag
#@TODO: min version instance
#@TODO: resume_from - hard to implement without saving all data and configurations


# implement save_best renaming
if __name__ == "__main__":
    sly.main_wrapper("main", main)
