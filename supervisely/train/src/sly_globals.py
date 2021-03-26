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


root_source_path = str(Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)


def init_project_info_and_meta():
    global project_info, project_meta
    project_info = api.project.get_info_by_id(project_id)
    project_meta_json = api.project.get_meta(project_id)
    project_meta = sly.ProjectMeta.from_json(project_meta_json)
