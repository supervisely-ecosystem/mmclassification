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


#@TODO: preview augentations
#@TODO: random weights initialization?
if __name__ == "__main__":
    sly.main_wrapper("main", main)
