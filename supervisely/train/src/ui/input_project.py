import os
from collections import defaultdict
import supervisely_lib as sly
import sly_globals as g
from sly_train_progress import get_progress_cb

images_infos = None # dataset_name -> image_name -> image_info


def init(data):
    data["projectId"] = g.project_info.id
    data["projectName"] = g.project_info.name
    data["projectImagesCount"] = g.project_info.items_count
    data["projectPreviewUrl"] = g.api.image.preview_url(g.project_info.reference_image_url, 100, 100)


def cache_images_infos():
    global images_infos
    images_infos = {}
    for dataset_info in g.api.dataset.get_list(g.project_id):
        images_infos[dataset_info.name] = {}
        for image_info in g.api.image.get_list(dataset_info.id):



def download():
    # if-else only to speedup debug, has no effect in prod
    if sly.fs.dir_exists(g.project_dir):
        pass
    else:
        sly.fs.mkdir(g.project_dir, remove_content_if_exists=False)  # clean content for debug, has no effect in prod
        # download and preprocess Sypervisely project (using cache)
        download_progress = get_progress_cb("Download data (using cache)", g.project_info.items_count * 2)
        sly.download_project(g.api, g.project_id, g.project_dir,
                             cache=g.my_app.cache, progress_cb=download_progress, only_image_tags=True)

# def clean_sets_and_calc_stats(project_dir, train_set, val_set, progress_cb):
#     project = sly.Project(project_dir, sly.OpenMode.READ)
#     train_tags = sly.json.load_json_file(os.path.join(project_dir, "gt_labels.json"))
#
#     def _clean_and_calc(split, stats):
#         res_split = []
#         for item in split:
#             ann = sly.Annotation.load_json_file(item.ann_path, project.meta)
#
#             name = None
#             num_train_tags_on_image = 0
#             for tag in ann.img_tags:
#                 tag: sly.Tag
#                 if tag.name in train_tags:
#                     name = tag.name
#                     num_train_tags_on_image += 1
#
#             if num_training_tags_on_image == 0:
#                 stats["no tags"] += 1
#             elif num_training_tags_on_image > 1:
#                 stats["collision"] += 1
#             else:
#                 if name is None:
#                     raise RuntimeError("Tag name is None")
#                 stats[name] += 1
#
#
#     stats = defaultdict(int)
#
#
#
#     num_images_not_tags = 0
#     num_images_multiple_tags = 0
#
#     to_remove = {}
#     for dataset in project.datasets:
#         to_remove[dataset.name] = {}
#         for item_name in dataset:
#             img_path, ann_path = dataset.get_item_paths(item_name)
#             ann = sly.Annotation.load_json_file(ann_path, project.meta)
#
#             num_training_tags_on_image = 0
#             for tag in ann.img_tags:
#                 tag: sly.Tag
#                 if tag.name in train_tags:
#                     num_training_tags_on_image += 1
#
#             if num_training_tags_on_image == 0:
#                 sly.logger.warn(f"Image {item_name} in dataset {dataset.name} does not have any any of the training tags, will be ignored")
#                 to_remove[dataset.name][item_name] = 1
#                 dataset.delete_item(item_name)  # to be sure that there are no bad images in training
#                 num_images_not_tags += 1
#             if num_training_tags_on_image > 1:
#                 sly.logger.warn(f"Conflict: multiple training tags were assigned to image {item_name} in dataset {dataset.name}, will be ignored")
#                 to_remove[dataset.name][item_name] = 1
#                 dataset.delete_item(item_name)  # to be sure that there are no bad images in training
#                 num_images_multiple_tags += 1
#
#             progress_cb(1)
#
#     def _filter(items, to_remove):
#         filtered = []
#         for item in items:
#             if item.name in to_remove[item.dataset_name]:
#                 continue  # remove item
#             filtered.append(item)
#         return filtered
#
#     filtered_train_set = _filter(train_set, to_remove)
#     filtered_val_set = _filter(val_set, to_remove)
#     return num_images_not_tags, num_images_multiple_tags, filtered_train_set, filtered_val_set