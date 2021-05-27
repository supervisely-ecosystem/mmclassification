import os
import supervisely_lib as sly
import sly_globals as g
import train_config


def init(data, state):
    state["epochs"] = 10
    state["gpusIds"] = '0'

    state["imgSize"] = 256
    state["batchSizePerGPU"] = 32
    state["workersPerGPU"] = 2  # 0 - for debug @TODO: for debug
    state["valInterval"] = 1
    state["metricsPeriod"] = 1

    state["optimizer"] = "SGD"
    state["lr"] = 0.001
    state["momentum"] = 0.9
    state["weightDecay"] = 0.0001
    state["nesterov"] = False
    state["gradClipEnabled"] = False
    state["maxNorm"] = 1

    state["lrPolicyEnabled"] = False

    file_path = os.path.join(g.root_source_dir, "configs/_base_/schedules/supervisely_lr_policy.py")
    with open(file_path) as f:
        state["lrPolicyPyConfig"] = f.read()

    state["metricsPeriod"] = 1
    state["valInterval"] = 1

    data["modelPyConfig"] = ""
    data["datasetPyConfig"] = ""
    data["schedulePyConfig"] = ""
    data["runtimePyConfig"] = ""
    data["mainPyConfig"] = ""

    #state["activeTabName"] = "General"


@g.my_app.callback("preview_configs")
@sly.timeit
def preview_configs(api: sly.Api, task_id, context, state, app_logger):
    model_config_path, model_py_config = train_config.generate_model_config(state)
    dataset_config_path, dataset_py_config = train_config.generate_dataset_config(state)
    schedule_config_path, schedule_py_config = train_config.generate_schedule_config(state)
    runtime_config_path, runtime_py_config = train_config.generate_runtime_config(state)
    main_config_path, main_py_config = train_config.generate_main_config(state)

    fields = [
        {"field": "data.modelPyConfig", "payload": model_py_config},
        {"field": "data.datasetPyConfig", "payload": dataset_py_config},
        {"field": "data.schedulePyConfig", "payload": schedule_py_config},
        {"field": "data.runtimePyConfig", "payload": runtime_py_config},
        {"field": "data.mainPyConfig", "payload": main_py_config},
    ]
    api.task.set_fields(task_id, fields)
