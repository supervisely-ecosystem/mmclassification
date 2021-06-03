import os
from collections import namedtuple
import shelve
import supervisely_lib as sly
import sly_globals as g
from sly_train_progress import get_progress_cb, reset_progress, init_progress

progress_index = 1
_images_infos = None # dataset_name -> image_name -> image_info
_cache_base_filename = os.path.join(g.my_app.data_dir, "images_info")
_cache_path = _cache_base_filename + ".db"


def init(data):
    data["projectId"] = g.project_info.id
    data["projectName"] = g.project_info.name
    data["projectImagesCount"] = g.project_info.items_count
    data["projectPreviewUrl"] = g.api.image.preview_url(g.project_info.reference_image_url, 100, 100)
    init_progress(progress_index, data)
    data["done1"] = False


@g.my_app.callback("download_project")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def download(api: sly.Api, task_id, context, state, app_logger):
    # *******************************************
    sly.fs.mkdir(g.project_dir) # for debug
    sly.fs.remove_dir(g.project_dir)  # for debug
    sly.fs.silent_remove(_cache_path) # for debug
    # *******************************************

    if sly.fs.dir_exists(g.project_dir):
        pass
    else:
        sly.fs.mkdir(g.project_dir)
        download_progress = get_progress_cb(progress_index, "Download project", g.project_info.items_count * 2)
        sly.download_project(g.api, g.project_id, g.project_dir,
                             cache=g.my_app.cache, progress_cb=download_progress, only_image_tags=True)
        reset_progress(progress_index)

    cache_images_infos()
    api.task.set_field(task_id, "data.done1", True)


def cache_images_infos():
    global _images_infos

    if sly.fs.file_exists(_cache_path):
        sly.logger.info("Cache exists, read images info from cache")
        with shelve.open(_cache_base_filename, flag='r') as s:
            _images_infos = s["images_infos"]
    else:
        progress = get_progress_cb(progress_index, "Cache project info", g.project_info.items_count)
        _images_infos = {}
        for dataset_info in g.api.dataset.get_list(g.project_id):
            name_to_info = {}
            for info in g.api.image.get_list(dataset_info.id):
                name_to_info[info.name] = info._asdict()
                progress(1)
            _images_infos[dataset_info.name] = name_to_info
        reset_progress(progress_index)

        with shelve.open(_cache_base_filename) as s:
            s["images_infos"] = _images_infos
        sly.logger.info(f"images infos for project id={g.project_id} has been successfully cached")


def get_image_info_from_cache(dataset_name, item_name):
    image_info_dict = _images_infos[dataset_name][item_name]
    ImageInfo = namedtuple('ImageInfo', image_info_dict)
    info = ImageInfo(**image_info_dict)
    return info



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