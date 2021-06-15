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
    #sly.fs.clean_dir(g.my_app.data_dir)
    g.my_app.run(data=data, state=state)

#@TODO: Error: TypeError("'<=' not supported between instances of 'NoneType' and 'int'")
# {"message": "please, contact support: task_id=5536, TypeError(\"'<=' not supported between instances of 'NoneType' and 'int'\")", "exc_str": "'<=' not supported between instances of 'NoneType' and 'int'", "timestamp": "2021-06-15T15:07:09.657Z", "level": "error", "stack": ["Traceback (most recent call last):", "  File \"/mmclassification/supervisely_lib/app/app_service.py\", line 394, in wrapper", "    f(*args, **kwargs)", "  File \"/mmclassification/supervisely/train/src/ui/ui.py\", line 39, in restart", "    if restart_from_step <= 2:", "TypeError: '<=' not supported between instances of 'NoneType' and 'int'"]}

#@TODO: tags cooccurance - добавить в readme, поменять там табличный виджет, сделать зафиксированные колонки и скрол горизонтальный
#@TODO: state["workersPerGPU"] = 0# 2  # 0 - for debug @TODO: for debug
#@TODO: add ON/OFF for custom augmentations
#@TODO: move mm_train import on top
#@TODO: custom weights - load-from option
#@TODO: readme - add py-configs to training artifacts
#@TODO: readme - tags co-occurrence-matrix
#@TODO: add to readme - unpack KV tag
#@TODO: resume_from - hard to implement without saving all data and configurations
if __name__ == "__main__":
    sly.main_wrapper("main", main)
