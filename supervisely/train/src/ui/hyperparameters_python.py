import train_config
import supervisely_lib as sly
import sly_globals as g


def init(data, state):
    state["modelPyConfig"] = ""
    state["datasetPyConfig"] = ""
    state["schedulePyConfig"] = ""
    state["runtimePyConfig"] = ""
    state["mainPyConfig"] = ""

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

    state["collapsed8"] = True
    state["disabled8"] = True
    state["done8"] = False


@g.my_app.callback("preview_configs")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def preview_configs(api: sly.Api, task_id, context, state, app_logger):
    model_config_path, model_py_config = train_config.generate_model_config(state)
    dataset_config_path, dataset_py_config = train_config.generate_dataset_config(state)
    schedule_config_path, schedule_py_config = train_config.generate_schedule_config(state)
    runtime_config_path, runtime_py_config = train_config.generate_runtime_config(state)
    main_config_path, main_py_config = train_config.generate_main_config(state)

    fields = [
        {"field": "state.modelPyConfig", "payload": model_py_config},
        {"field": "state.datasetPyConfig", "payload": dataset_py_config},
        {"field": "state.schedulePyConfig", "payload": schedule_py_config},
        {"field": "state.runtimePyConfig", "payload": runtime_py_config},
        {"field": "state.mainPyConfig", "payload": main_py_config},
    ]
    api.task.set_fields(task_id, fields)