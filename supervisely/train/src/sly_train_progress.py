from functools import partial
import os
import time
import math
import supervisely_lib as sly
import sly_globals as globals


def update_progress(count, index, api: sly.Api, task_id, progress: sly.Progress):
    progress.iters_done(count)
    _update_progress_ui(index, api, task_id, progress)


def _update_progress_ui(index, api: sly.Api, task_id, progress: sly.Progress, stdout_print=False):
    if progress.need_report():
        fields = [
            {"field": f"data.progress{index}", "payload": progress.message},
            {"field": f"data.progressCurrent{index}", "payload": progress.current_label},
            {"field": f"data.progressTotal{index}", "payload": progress.total_label},
            {"field": f"data.progressPercent{index}", "payload": math.floor(progress.current * 100 / progress.total)},
        ]
        api.app.set_fields(task_id, fields)
        # if stdout_print is True:
        #     #progress.print_progress()
        progress.report_progress()


def get_progress_cb(index, message, total, is_size=False):
    progress = sly.Progress(message, total, is_size=is_size)
    progress_cb = partial(update_progress, index=index, api=globals.api, task_id=globals.task_id, progress=progress)
    progress_cb(0)
    return progress_cb


def reset_progress(index):
    fields = [
        {"field": f"data.progress{index}", "payload": None},
        {"field": f"data.progressCurrent{index}", "payload": None},
        {"field": f"data.progressTotal{index}", "payload": None},
        {"field": f"data.progressPercent{index}", "payload": None},
    ]
    globals.api.app.set_fields(globals.task_id, fields)


def init_progress(index, data):
    data[f"progress{index}"] = None
    data[f"progressCurrent{index}"] = None
    data[f"progressTotal{index}"] = None
    data[f"progressPercent{index}"] = None


def update_uploading_progress(count, api: sly.Api, task_id, progress: sly.Progress):
    progress.iters_done(count - progress.current)
    _update_progress_ui(api, task_id, progress, stdout_print=True)