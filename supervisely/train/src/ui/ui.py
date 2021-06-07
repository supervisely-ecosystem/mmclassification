import supervisely_lib as sly
import sly_globals as g
import input_project as input_project
import tags
import splits as train_val_split
import validate_training_data
import augs
import architectures as model_architectures
import hyperparameters as hyperparameters
import hyperparameters_python as hyperparameters_python
import monitoring as monitoring
# import artifacts as artifacts


@sly.timeit
def init(data, state):
    state["activeStep"] = 1
    state["restartDialog"] = False
    input_project.init(data, state)
    tags.init(data, state)
    train_val_split.init(g.project_info, g.project_meta, data, state)
    validate_training_data.init(data, state)
    augs.init(data, state)
    model_architectures.init(data, state)
    hyperparameters.init(data, state)
    hyperparameters_python.init(data, state)
    monitoring.init(data, state)
    # artifacts.init(data)


@g.my_app.callback("restart")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def restart(api: sly.Api, task_id, context, state, app_logger):
    pass
