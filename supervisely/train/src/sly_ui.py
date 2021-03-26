import os
import supervisely_lib as sly

import sly_globals as globals
#import sly_metrics as metrics


# empty_gallery = {
#     "content": {
#         "projectMeta": sly.ProjectMeta().to_json(),
#         "annotations": {},
#         "layout": []
#     }
# }


def init_input_project(data, project_info):
    data["projectId"] = globals.project_id
    data["projectName"] = project_info.name
    data["projectImagesCount"] = project_info.items_count
    data["projectPreviewUrl"] = globals.api.image.preview_url(project_info.reference_image_url, 100, 100)


def init_tags_stats(data, state, project_meta: sly.ProjectMeta):
    stats = globals.api.project.get_stats(globals.project_id)
    images_with_tag = {}
    for item in stats["imageTags"]["items"]:
        images_with_tag[item["tagMeta"]["name"]] = item["total"]

    tags_json = []

    for tag_meta in project_meta.tag_metas.to_json():
        tag_meta["imagesCount"] = images_with_tag[tag_meta["name"]]
        tag_meta["valueType"] = tag_meta["value_type"]
        if tag_meta["valueType"] != sly.TagValueType.NONE or tag_meta["name"] in ["train", "val"]:
            continue
        tags_json.append(tag_meta)

    data["tags"] = tags_json
    state["selectedTags"] = []
    #state["tags"] = len(tags_json) * [True]

#
# def init_random_split(PROJECT, data, state):
#     data["randomSplit"] = [
#         {"name": "train", "type": "success"},
#         {"name": "val", "type": "primary"},
#         {"name": "total", "type": "gray"},
#     ]
#     data["totalImagesCount"] = PROJECT.items_count
#
#     train_percent = 80
#     train_count = int(PROJECT.items_count / 100 * train_percent)
#     state["randomSplit"] = {
#         "count": {
#             "total": PROJECT.items_count,
#             "train": train_count,
#             "val": PROJECT.items_count - train_count
#         },
#         "percent": {
#             "total": 100,
#             "train": train_percent,
#             "val": 100 - train_percent
#         },
#         "shareImagesBetweenSplits": False,
#         "sliderDisabled": False,
#     }
#
#     state["splitMethod"] = 1
#     state["trainTagName"] = ""
#     state["valTagName"] = ""
#
#
# def init_model_settings(data, state):
#     data["modelSizes"] = [
#         {"label": "yolov5s", "config": "yolov5s.yaml", "params": "7.3M"},
#         {"label": "yolov5m", "config": "yolov5m.yaml", "params": "21.4M"},
#         {"label": "yolov5l", "config": "yolov5l.yaml", "params": "47.0M"},
#         {"label": "yolov5x", "config": "yolov5x.yaml", "params": "87.7M"},
#     ]
#     state["modelSize"] = data["modelSizes"][0]["label"]
#     state["modelWeightsOptions"] = 1
#     state["pretrainedWeights"] = f'{data["modelSizes"][0]["label"]}.pt'
#
#     # @TODO: for debug
#     state["weightsPath"] = "/yolov5_train/coco128_002/2390/weights/best.pt"
#     #state["weightsPath"] = ""
#
#
# def init_training_hyperparameters(state):
#     state["epochs"] = 10
#     state["batchSize"] = 16
#     state["imgSize"] = 640
#     state["multiScale"] = False
#     state["singleClass"] = False
#     state["device"] = '0'
#     state["workers"] = 8 # 0 - for debug
#     state["activeTabName"] = "General"
#     state["hyp"] = {
#         "scratch": globals.scratch_str,
#         "finetune": globals.finetune_str,
#     }
#     state["hypRadio"] = "scratch"
#
#
# def init_start_state(state):
#     state["started"] = False
#     state["activeNames"] = []
#
#
# def init_galleries(data):
#     data["vis"] = empty_gallery
#     data["labelsVis"] = empty_gallery
#     data["predVis"] = empty_gallery
#     data["syncBindings"] = []
#
#
# def init_progress(data):
#     data["progressName"] = ""
#     data["currentProgress"] = 0
#     data["totalProgress"] = 0
#     data["currentProgressLabel"] = ""
#     data["totalProgressLabel"] = ""
#
#
# def init_output(data):
#     data["outputUrl"] = ""
#     data["outputName"] = ""


def init(data, state):
    init_input_project(data, globals.project_info)
    init_tags_stats(data, state, globals.project_meta)
    # init_random_split(globals.project_info, data, state)
    # init_model_settings(data, state)
    # init_training_hyperparameters(state)
    # init_start_state(state)
    # init_galleries(data)
    # init_progress(data)
    # init_output(data)
    #metrics.init(data, state)

#
# def set_output():
#     file_info = globals.api.file.get_info_by_path(globals.team_id,
#                                                   os.path.join(globals.remote_artifacts_dir, 'results.png'))
#     fields = [
#         {"field": "data.outputUrl", "payload": globals.api.file.get_url(file_info.id)},
#         {"field": "data.outputName", "payload": globals.remote_artifacts_dir},
#     ]
#     globals.api.app.set_fields(globals.task_id, fields)
#     globals.api.task.set_output_directory(globals.task_id, file_info.id, globals.remote_artifacts_dir)

