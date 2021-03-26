import os
import supervisely_lib as sly
import sly_ui as ui
import sly_globals as g


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "modal.state.slyProjectId": g.project_id,
    })

    data = {}
    state = {}
    data["taskId"] = g.task_id

    # read project information and meta (classes + tags)
    g.init_project_info_and_meta()

    # init data for UI widgets
    ui.init(data, state)

    g.app.run(data=data, state=state)


if __name__ == "__main__":
    #sly.main_wrapper("main", main)
    main() # for debug