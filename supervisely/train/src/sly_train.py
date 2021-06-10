import os
import supervisely_lib as sly
import sly_globals as g
import ui as ui
from sly_train_progress import get_progress_cb
from sly_train_args import init_script_arguments
import sly_logger_hook


@g.my_app.callback("train")
@sly.timeit
#@g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    #try:
        # init sys.argv for main training script
    init_script_arguments(state)
    from tools.train import main as mm_train #@TODO: move to imports section on top
    mm_train()

        # # upload artifacts directory to Team Files
        # upload_artifacts(g.local_artifacts_dir, g.remote_artifacts_dir)
        # set_task_output()
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
