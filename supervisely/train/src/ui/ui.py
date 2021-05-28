import supervisely_lib as sly
import sly_globals as g
import input_project as input_project
import tags
import splits as train_val_split
import augs
import architectures as model_architectures
import hyperparameters as hyperparameters
import hyperparameters_python as hyperparameters_python
import monitoring as monitoring
# import artifacts as artifacts


@sly.timeit
def init(data, state):
    input_project.init(data)
    tags.init(data, state)
    train_val_split.init(g.project_info, g.project_meta, data, state)
    augs.init(data, state)
    model_architectures.init(data, state)
    hyperparameters.init(data, state)
    hyperparameters_python.init(data, state)
    monitoring.init(data, state)
    # artifacts.init(data)
