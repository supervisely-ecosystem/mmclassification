import os
import re
import supervisely_lib as sly

import sly_globals as g
import architectures


def generate_model_config(configs_dir, state):
    model_name = state["selectedModel"]
    model_info = architectures.get_model_info_by_name(model_name)
    model_config_path = os.path.join(g.root_source_dir, model_info["modelConfig"])
    with open(model_config_path) as f:
        py_config = f.read()

    regex = r"num_classes*=(.*),"
    num_tags = len(state["selectedTags"])
    def _my_replace_function(match):
        #to_replace = match.group(0)
        #part_path = match.group(1)
        return f"num_classes = {num_tags},"
    result = re.sub(regex, lambda m: _my_replace_function(m), py_config, 0, re.MULTILINE)

    config_path = os.path.join(configs_dir, f"{sly.fs.get_file_name(model_config_path)}_sly.py")
    with open(config_path, 'w') as f:
        f.write(result)


def generate(state):
    configs_dir = os.path.join(g.my_app.data_dir, "configs")
    sly.fs.mkdir(configs_dir)
    generate_model_config(configs_dir, state)

    res_config_path = os.path.join(g.my_app.data_dir, "train_config.py")

# def generate_config(save_path):
#     pass
#
# def generate(model_config_path, save_path):
#     import importlib
#     spec = importlib.util.spec_from_file_location("model_config", model_config_path)
#     foo = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(foo)
#     print(foo._base_)