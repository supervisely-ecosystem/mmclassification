from functools import partial
import os
import time
import math
import supervisely as sly
import sly_globals as globals


def update_progress(count, index, api: sly.Api, task_id, progress: sly.Progress):
    progress.iters_done(count)
    _update_progress_ui(index, api, task_id, progress)


def set_progress(count, index, api: sly.Api, task_id, progress: sly.Progress):
    progress.set_current_value(count)
    _update_progress_ui(index, api, task_id, progress)


def _update_progress_ui(index, api: sly.Api, task_id, progress: sly.Progress, stdout_print=False):
    if progress.need_report():
        # Fix for displaying bytes instead of file count
        current_label = progress.current_label
        total_label = progress.total_label
        
        # If this is a size-based progress (showing bytes), convert to more readable format
        if progress.is_size and isinstance(progress.total, int) and progress.total > 1000:
            # Format as megabytes if total is large enough
            current_mb = progress.current / (1024 * 1024)
            total_mb = progress.total / (1024 * 1024)
            current_label = f"{current_mb:.2f} MB"
            total_label = f"{total_mb:.2f} MB"
        
        # Log detailed progress info for debugging
        sly.logger.debug(
            f"Progress update for {index}", 
            extra={
                "current": progress.current,
                "total": progress.total,
                "current_label": current_label,
                "total_label": total_label,
                "is_size": progress.is_size
            }
        )
            
        fields = [
            {"field": f"data.progress{index}", "payload": progress.message},
            {"field": f"data.progressCurrent{index}", "payload": current_label},
            {"field": f"data.progressTotal{index}", "payload": total_label},
            {"field": f"data.progressPercent{index}", "payload": math.floor(progress.current * 100 / max(progress.total, 1))},
        ]
        api.app.set_fields(task_id, fields)
        progress.report_progress()


def get_progress_cb(index, message, total, is_size=False, min_report_percent=5, upd_func=update_progress):
    # Ensure we have a reasonable value for total if tracking files
    if not is_size and (total is None or total <= 0):
        total = 100  # Default to 100 if total is invalid
        sly.logger.warning(f"Invalid total for progress {index} ({message}), using default value: {total}")
    
    progress = sly.Progress(message, total, is_size=is_size, min_report_percent=min_report_percent)
    
    # Log progress creation
    sly.logger.debug(
        f"Created progress for {index}", 
        extra={
            "message": message,
            "total": total,
            "is_size": is_size
        }
    )
    
    progress_cb = partial(upd_func, index=index, api=globals.api, task_id=globals.task_id, progress=progress)
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
    _update_progress_ui(0, api, task_id, progress, stdout_print=True)  # Fixed missing index parameter


def add_progress_to_request(fields, index, progress: sly.Progress):
    if progress is None:
        res = [
            {"field": f"data.progress{index}", "payload": None},
            {"field": f"data.progressCurrent{index}", "payload": None},
            {"field": f"data.progressTotal{index}", "payload": None},
            {"field": f"data.progressPercent{index}", "payload": None},
        ]
    else:
        # Format labels for better readability, especially for file sizes
        current_label = progress.current_label
        total_label = progress.total_label
        
        # Convert to MB for better readability when tracking bytes
        if progress.is_size and isinstance(progress.total, int) and progress.total > 1000:
            current_mb = progress.current / (1024 * 1024)
            total_mb = progress.total / (1024 * 1024)
            current_label = f"{current_mb:.2f} MB"
            total_label = f"{total_mb:.2f} MB"
            
        res = [
            {"field": f"data.progress{index}", "payload": progress.message},
            {"field": f"data.progressCurrent{index}", "payload": current_label},
            {"field": f"data.progressTotal{index}", "payload": total_label},
            {"field": f"data.progressPercent{index}", "payload": math.floor(progress.current * 100 / max(progress.total, 1))},
        ]
    fields.extend(res)