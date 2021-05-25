import os
import sly_globals as g
import architectures
import re


def generate_model_config(state):
    model_name = state["selectedModel"]
    model_info = architectures.get_model_info_by_name(model_name)
    model_config_path = os.path.join(g.root_source_dir, model_info["modelConfig"])

    with open(model_config_path) as f:
        py_config = f.read()

    print(py_config)

    regex = r"{\%.*include.*'(.*)'.*\%}"
    regex = r"num_classes*=(.*),"  #num_classes = 1000,

    def _my_replace_function(match):
        to_replace = match.group(0)
        print(to_replace)
        part_path = match.group(1)
        print(part_path)
        return 777
    result = re.sub(regex, lambda m: _my_replace_function(m), py_config, 0, re.MULTILINE)

    x = 10
    x += 1


def generate(state):
    generate_model_config(state)

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