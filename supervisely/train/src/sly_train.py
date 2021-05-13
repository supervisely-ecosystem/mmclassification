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

    img_info = g.api.image.get_info_by_id(859567)
    id_to_tagmeta = g.project_meta.tag_metas.get_id_mapping()
    print(img_info.tags)
    tags = sly.TagCollection.from_api_response(img_info.tags, g.project_meta.tag_metas, id_to_tagmeta)
    print(tags)

    g.my_app.compile_template(g.root_source_dir)

    data = {}
    state = {}
    data["taskId"] = g.task_id
    ui.init(data, state)  # init data for UI widgets

    g.my_app.run(data=data, state=state)


#@TODO: check image tags in info
#@TODO: preview augentations
#@TODO: random weights initialization?
if __name__ == "__main__":
    sly.main_wrapper("main", main)
