import train_config
import supervisely_lib as sly
import sly_globals as g


def init(data, state):
    data["modelPyConfig"] = ""
    data["datasetPyConfig"] = ""
    data["schedulePyConfig"] = ""
    data["runtimePyConfig"] = ""
    data["mainPyConfig"] = ""

    data["configsPyViewOptionsRead"] = {
        "mode": 'ace/mode/python',
        "showGutter": False,
        "readOnly": True,
        "maxLines": 100,
        "highlightActiveLine": False
    }

    data["configsPyViewOptionsWrite"] = {
        "mode": 'ace/mode/python',
        "showGutter": True,
        "readOnly": False,
        "maxLines": 100,
        "highlightActiveLine": True
    }

    state["pyConfigsViewOptions"] = data["configsPyViewOptionsRead"]

    state["advancedPy"] = False


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