import os
import supervisely as sly
import sly_globals as g


def init(data, state):
    state["epochs"] = 5
    state["gpusId"] = '0'

    state["imgSize"] = 224
    state["batchSizePerGPU"] = 32
    state["workersPerGPU"] = 2  #@TODO: 0 - for debug
    state["valInterval"] = 1
    state["metricsPeriod"] = 10
    state["checkpointInterval"] = 1
    state["maxKeepCkptsEnabled"] = True
    state["maxKeepCkpts"] = 3
    state["saveLast"] = True
    state["saveBest"] = True
    state["disabledImgSize"] = False

    state["optimizer"] = "SGD"
    state["lr"] = 0.001
    state["momentum"] = 0.9
    state["weightDecay"] = 0.0001
    state["nesterov"] = False
    state["gradClipEnabled"] = False
    state["maxNorm"] = 1

    state["lrPolicyEnabled"] = False

    file_path = os.path.join(g.root_source_dir, "supervisely/train/configs/lr_policy.py")
    with open(file_path) as f:
        state["lrPolicyPyConfig"] = f.read()

    state["collapsed7"] = True
    state["disabled7"] = True
    data["done7"] = False


def restart(data, state):
    data["done7"] = False


@g.my_app.callback("use_hyp")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_hyp(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "data.done7", "payload": True},
        {"field": "state.collapsed8", "payload": False},
        {"field": "state.disabled8", "payload": False},
        {"field": "state.activeStep", "payload": 8},
    ]
    g.api.app.set_fields(g.task_id, fields)
