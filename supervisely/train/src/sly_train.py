import os
import supervisely_lib as sly

import sly_globals as g
import ui as ui


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "modal.state.slyProjectId": g.project_id,
    })

    g.my_app.compile_template(g.root_source_dir)

    data = {}
    state = {}
    data["taskId"] = g.task_id
    ui.init(data, state)  # init data for UI widgets

    g.my_app.run(data=data, state=state)
    sly.upload_project()

#@TODO: custom weights - load-from option
#@ __todo_ base models - '../_base_/models/resnet50_supe.py'
# num_classes=XXX
# topk=(1, 5), - option
# backbone=dict(type='ResNet_CIFAR'
#

#@TODO: add predicted tags to model file
#@TODO: separate - update content and options in comparegallery
#@TODO: disable preview button if custom pipeline is not defined
#@TODO: augs templates
#@TODO: preview augentations
#@TODO: random weights initialization?
#@TODO: --resume-from - continue training
if __name__ == "__main__":
    sly.main_wrapper("main", main)
