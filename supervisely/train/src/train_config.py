import os
import re
import supervisely_lib as sly

import sly_globals as g
import architectures

model_config_name = "model_config.py"
dataset_config_name = "dataset_config.py"
schedule_config_name = "schedule_config.py"
runtime_config_name = "runtime_config.py"
train_config_name = "train_config.py"
configs_dir = os.path.join(g.my_app.data_dir, "configs")
sly.fs.mkdir(configs_dir)
sly.fs.clean_dir(configs_dir)  # for debug


def _replace_function(var_name, var_value, template, match):
    m0 = match.group(0)
    m1 = match.group(1)
    return template.format(var_name, var_value)


def generate_model_config(state):
    model_name = state["selectedModel"]
    model_info = architectures.get_model_info_by_name(model_name)
    model_config_path = os.path.join(g.root_source_dir, model_info["modelConfig"])
    with open(model_config_path) as f:
        py_config = f.read()

    num_tags = len(state["selectedTags"])
    py_config = re.sub(r"num_classes*=(\d+),",
                       lambda m: _replace_function("num_classes", num_tags, "{}={},", m),
                       py_config, 0, re.MULTILINE)

    config_path = os.path.join(configs_dir, model_config_name)
    with open(config_path, 'w') as f:
        f.write(py_config)
    return config_path, py_config


def generate_dataset_config(state):
    config_path = os.path.join(g.root_source_dir, "configs/_base_/datasets/supervisely.py")
    with open(config_path) as f:
        py_config = f.read()

    py_config = re.sub(r"input_size\s*=\s*(\d+)",
                       lambda m: _replace_function("input_size", state["imgSize"], "{} = {}", m),
                       py_config, 0, re.MULTILINE)

    py_config = re.sub(r"batch_size_per_gpu\s*=\s*(\d+)",
                       lambda m: _replace_function("batch_size_per_gpu", state["batchSizePerGPU"], "{} = {}", m),
                       py_config, 0, re.MULTILINE)

    py_config = re.sub(r"num_workers_per_gpu\s*=\s*(\d+)",
                       lambda m: _replace_function("workersPerGPU", state["workersPerGPU"], "{} = {}", m),
                       py_config, 0, re.MULTILINE)

    py_config = re.sub(r"validation_interval\s*=\s*(\d+)",
                       lambda m: _replace_function("validation_interval", state["valInterval"], "{} = {}", m),
                       py_config, 0, re.MULTILINE)

    config_path = os.path.join(configs_dir, dataset_config_name)
    with open(config_path, 'w') as f:
        f.write(py_config)
    return config_path, py_config


def generate_schedule_config(state):
    pass


def generate(state):
    raise NotImplementedError()
    #model_config_path, model_py_config = generate_model_config(state)
    #dataset_config_path, model_py_config = generate_dataset_config(state)


    #res_config_path = os.path.join(g.my_app.data_dir, "train_config.py")

# def generate_config(save_path):
#     pass
#
# def generate(model_config_path, save_path):
#     import importlib
#     spec = importlib.util.spec_from_file_location("model_config", model_config_path)
#     foo = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(foo)
#     print(foo._base_)