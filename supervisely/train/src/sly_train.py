import os
import supervisely_lib as sly
import sly_globals as g
import ui as ui
import sly_logger_hook  # to register hook
import sly_imgaugs  # to register first part of the pipeline


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
    #sly.fs.clean_dir(g.my_app.data_dir) #@TODO: for debug
    g.my_app.run(data=data, state=state)


#@TODO: tags cooccurance - добавить в readme, поменять там табличный виджет, сделать зафиксированные колонки и скрол горизонтальный
#@TODO: move mm_train import on top
#@TODO: custom weights - load-from option
#@TODO: readme - describe all training artifacts
#@TODO: add to readme - unpack KV tag
#@TODO: resume_from - hard to implement without saving all data and configurations
if __name__ == "__main__":
    sly.main_wrapper("main", main)
