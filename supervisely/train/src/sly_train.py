import os
import supervisely_lib as sly
import sly_globals as g
import ui as ui
import sly_logger_hook  # to register hook


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
    #sly.fs.clean_dir(g.my_app.data_dir)
    g.my_app.run(data=data, state=state)


#@TODO: handle training errors
#@TODO: save references
#@TODO: custom augs - reimplement prepare data in BaseDataset

#@TODO: tags cooccurance - добавить в readme, поменять там табличный виджет, сделать зафиксированные колонки и скрол горизонтальный
#@TODO: save session link in artifacts dir
#@TODO: state["workersPerGPU"] = 0# 2  # 0 - for debug @TODO: for debug
#@TODO: add ON/OFF for custom augmentations

#@TODO: move mm_train import on top
#@TODO: release new version of SDK before release app
#@TODO: if OOM error, make a special message for that
#@TODO: custom weights - load-from option
#@TODO: readme - add py-configs to training artifacts
#@TODO: readme - tags co-occurrence-matrix
#@TODO: add to readme - unpack KV tag
#@TODO: min version instance
#@TODO: resume_from - hard to implement without saving all data and configurations
if __name__ == "__main__":
    sly.main_wrapper("main", main)
