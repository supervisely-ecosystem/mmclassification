import os
import supervisely_lib as sly
import sly_globals as g

_templates = [
    {
        "config": "supervisely/train/augs/mmclass-light.json",
        "name": "Light (color + rotate)",
    },
    {
        "config": "supervisely/train/augs/mmclass-light-with-fliplr.json",
        "name": "Light + fliplr",
    },
    {
        "config": "supervisely/train/augs/mmclass-heavy-no-fliplr.json",
        "name": "Heavy",
    },
    {
        "config": "supervisely/train/augs/mmclass-heavy-with-fliplr.json",
        "name": "Heavy + fliplr",
    },
]


def _load_template(json_path):
    config = sly.json.load_json_file(json_path)
    pipeline = sly.imgaug_utils.build_pipeline(config["pipeline"], random_order=config["random_order"])  # to validate
    py_code = sly.imgaug_utils.pipeline_to_python(config["pipeline"], config["random_order"])
    return pipeline, py_code


def get_aug_templates_list():
    pipelines_info = []
    name_to_py = {}
    for template in _templates:
        json_path = os.path.join(g.root_source_dir, template["config"])
        _, py_code = _load_template(json_path)
        pipelines_info.append({
            **template,
            "py": py_code
        })
        name_to_py[template["name"]] = py_code
    return pipelines_info, name_to_py


def get_template_by_name(name):
    for template in _templates:
        if template["name"] == name:
            json_path = os.path.join(g.root_source_dir, template["config"])
            pipeline, _ = _load_template(json_path)
            return pipeline
    raise KeyError(f"Template \"{name}\" not found")


def init(data, state):
    state["augsType"] = "template"
    templates_info, name_to_py = get_aug_templates_list()
    data["augTemplates"] = templates_info
    data["augPythonCode"] = name_to_py
    state["augsTemplateName"] = templates_info[0]["name"]

    data["pyViewOptions"] = {
        "mode": 'ace/mode/python',
        "showGutter": False,
        "readOnly": True,
        "maxLines": 100,
        "highlightActiveLine": False
    }
