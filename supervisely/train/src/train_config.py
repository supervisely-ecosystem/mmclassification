import os
import re
import supervisely_lib as sly

import sly_globals as g
import architectures

model_config_name = "model_config.py"
dataset_config_name = "dataset_config.py"
schedule_config_name = "schedule_config.py"
runtime_config_name = "runtime_config.py"
main_config_name = "train_config.py"
main_config_template = f"""
_base_ = [
    './{model_config_name}', './{dataset_config_name}',
    './{schedule_config_name}', './{runtime_config_name}'
]
"""

configs_dir = os.path.join(g.artifacts_dir, "configs")
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

    #https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/hooks/evaluation.html#EvalHook
    save_best = None if state["saveBest"] is False else "'auto'"
    py_config = re.sub(r"save_best\s*=\s*([a-zA-Z]+)\s",
                       lambda m: _replace_function("save_best", save_best, "{} = {}\n", m),
                       py_config, 0, re.MULTILINE)

    config_path = os.path.join(configs_dir, dataset_config_name)
    with open(config_path, 'w') as f:
        f.write(py_config)
    return config_path, py_config


def generate_schedule_config(state):
    optimizer = f"optimizer = dict(type='{state['optimizer']}', " \
                f"lr={state['lr']}, " \
                f"momentum={state['momentum']}, " \
                f"weight_decay={state['weightDecay']}" \
                f"{', nesterov=True' if (state['nesterov'] is True and state.optimizer == 'SGD') else ''})"

    grad_clip = f"optimizer_config = dict(grad_clip=None)"
    if state["gradClipEnabled"] is True:
        grad_clip = f"optimizer_config = dict(grad_clip=dict(max_norm={state['maxNorm']}))"

    ls_updater = ""
    if state["lrPolicyEnabled"] is True:
        py_text = state["lrPolicyPyConfig"]
        py_lines = py_text.splitlines()
        num_uncommented = 0
        for line in py_lines:
            res_line = line.strip()
            if res_line != "" and res_line[0] != "#":
                ls_updater += res_line
                num_uncommented += 1
        if num_uncommented == 0:
            raise ValueError("LR policy is enabled but not defined, please uncomment and modify one of the provided examples")
        if num_uncommented > 1:
            raise ValueError("several LR policies were uncommented, please keep only one")

    runner = f"runner = dict(type='EpochBasedRunner', max_epochs={state['epochs']})"

    # https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/hooks/checkpoint.html
    add_ckpt_to_config = []
    def _get_ckpt_arg(arg_name, state_flag, state_field, suffix=","):
        flag = True if state_flag is None else state[state_flag]
        if flag is True:
            add_ckpt_to_config.append(True)
            return f" {arg_name}={state[state_field]}{suffix}"
        return ""
    checkpoint = "checkpoint_config = dict({interval}{max_keep_ckpts}{save_last})".format(
        interval=_get_ckpt_arg("interval", None, "checkpointInterval"),
        max_keep_ckpts=_get_ckpt_arg("max_keep_ckpts", "maxKeepCkptsEnabled", "maxKeepCkpts"),
        save_last=_get_ckpt_arg("save_last", "saveLast", "saveLast", suffix=""),
    )

    py_config = optimizer + os.linesep + \
                grad_clip + os.linesep + \
                ls_updater + os.linesep + \
                runner + os.linesep
    if len(add_ckpt_to_config) > 0:
        py_config += checkpoint + os.linesep

    config_path = os.path.join(configs_dir, schedule_config_name)
    with open(config_path, 'w') as f:
        f.write(py_config)
    return config_path, py_config


def generate_runtime_config(state):
    return "", ""


def generate_main_config(state):
    config_path = os.path.join(configs_dir, main_config_name)
    with open(config_path, 'w') as f:
        f.write(main_config_template)
    return config_path, str(main_config_template)


def save_from_state(state):
