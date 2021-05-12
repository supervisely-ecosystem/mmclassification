import os
from pathlib import Path
import sys
import yaml
import supervisely_lib as sly

app = sly.AppService()
team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
project_id = int(os.environ['modal.state.slyProjectId'])

api: sly.Api = app.public_api
task_id = app.task_id

local_artifacts_dir = None
remote_artifacts_dir = None

project_info = None
project_meta = None

models_info = sly.json.load_json_file("models.json")
model_info_by_name = {info["model"]: info for info in models_info}

root_source_path = str(Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)


augs_json = None


def init_project_info_and_meta():
    global project_info, project_meta
    project_info = api.project.get_info_by_id(project_id)
    sly.logger.debug("Project info", extra={"project": project_info._asdict()})
    project_meta_json = api.project.get_meta(project_id)
    project_meta = sly.ProjectMeta.from_json(project_meta_json)


def read_text_from_file(path):
    with open(os.path.join(root_source_path, path), 'r') as file:
        data = file.read()
    return data


def init_default_augs():
    augs_path = os.path.join(root_source_path, "supervisely/train/augs/default_01.json")
    augs_config = sly.json.load_json_file(augs_path)
    augs = sly.aug.load_imgaug(augs_config)
    x = 10
