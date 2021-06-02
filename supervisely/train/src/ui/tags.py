import os
from collections import defaultdict, namedtuple
import shelve

import input_project
import supervisely_lib as sly
import random
import splits
import sly_globals as g
from sly_train_progress import get_progress_cb, reset_progress, init_progress

tag2images = None
tag2urls = None
images_without_tags = []

progress_index = 3
_preview_height = 120
_max_examples_count = 10

_ignore_tags = ["train", "val"]
_allowed_tag_types = [sly.TagValueType.NONE]

image_slider_options = {
    "selectable": False,
    "height": f"{_preview_height}px"
}

# speedup during debug (has no effects in production)
cache_base_filename = os.path.join(g.my_app.data_dir, f"{g.project_id}")
cache_path = cache_base_filename + ".db"


def init(data, state):
    # cache_images_examples(data)
    #
    # max_count = -1
    # tags_balance_rows = []
    # # tags with 0 images will be ignored automatically
    # for tag_name, images_infos in tag2images.items():
    #     if tag_name.lower() in _ignore_tags:
    #         continue
    #     tag_meta = g.project_meta.get_tag_meta(tag_name)
    #     tag_meta: sly.TagMeta
    #     if tag_meta.value_type not in _allowed_tag_types:
    #         continue
    #     tags_balance_rows.append({
    #         "name": tag_name,
    #         "total": len(images_infos),
    #         "segments": {
    #             "count": len(images_infos)
    #         }
    #     })
    #     max_count = max(max_count, len(images_infos))
    #
    # tags_balance = {
    #     "maxValue": max_count,
    #     "segments": [{"name": "Images count", "key": "count", "color": "#1892f8"}],
    #     "rows": tags_balance_rows
    # }

    data["tagsBalance"] = None  # tags_balance
    state["selectedTags"] = []
    state["tagsInProgress"] = False
    # stats = g.api.project.get_stats(g.project_id)
    # images_with_tag = {}
    # for item in stats["imageTags"]["items"]:
    #     images_with_tag[item["tagMeta"]["name"]] = item["total"]
    #
    # tags_json = []
    # for tag_meta in g.project_meta.tag_metas.to_json():
    #     tag_meta["imagesCount"] = images_with_tag[tag_meta["name"]]
    #     tag_meta["valueType"] = tag_meta["value_type"]
    #     if tag_meta["valueType"] != sly.TagValueType.NONE or tag_meta["name"] in ["train", "val"]:
    #         continue
    #     tags_json.append(tag_meta)
    # data["tags"] = tags_json
    # cache_images_examples(data)
    data["imageSliderOptions"] = image_slider_options
    data["done3"] = False
    init_progress(progress_index, data)


# def cache_images_examples(data):
#     global tag2urls, tag2images
#
#     if sly.fs.file_exists(cache_path):
#         sly.logger.info("Cache exists, read tags and images info from cache")
#         with shelve.open(cache_base_filename, flag='r') as s:
#             tag2urls = s["tag2urls"]
#             tag2images = s["tag2images"]
#     else:
#         temp_tag2images = g.api.project.download_images_tags(g.project_id)
#         for tag_name, images_infos in temp_tag2images.items():
#             #infos_dict = []
#             urls_examples = []
#             for info in images_infos:
#                 #infos_dict.append(info._asdict())
#                 if len(urls_examples) < _max_examples_count:
#                     urls_examples.append({
#                         "moreExamples": [info.full_storage_url],
#                         "preview": g.api.image.preview_url(info.full_storage_url, height=_preview_height)
#                     })
#
#             tag2images[tag_name] = images_infos  # infos_dict
#             tag2urls[tag_name] = urls_examples
#
#         with shelve.open(cache_base_filename) as s:
#             s["tag2urls"] = tag2urls
#             s["tag2images"] = tag2images
#         sly.logger.info(f"Cache for project id={g.project_id} has been successfully created")
#
#     data["tag2urls"] = tag2urls
#     #data["tag2images"] = tag2images


# def get_random_image():
#     rand_key = random.choice(list(tag2images.keys()))
#     image_info_dict = random.choice(tag2images[rand_key])
#     ImageInfo = namedtuple('ImageInfo', image_info_dict)
#     info = ImageInfo(**image_info_dict)
#     return info


def init_cache(split_items, split_name, progress_cb):
    global tag2images, tag2urls
    for item in split_items:
        name = item.name
        dataset_name = item.dataset_name
        ann_path = item.ann_path
        img_info = input_project.get_image_info_from_cache(dataset_name, name)

        ann = sly.Annotation.load_json_file(ann_path, g.project_meta)
        if len(ann.img_tags) == 0:
            images_without_tags.append(img_info)
        else:
            for tag in ann.img_tags:
                tag2images[tag.name][split_name].append(img_info)
                tag2urls[tag.name].append({
                    "moreExamples": [img_info.full_storage_url],
                    "preview": g.api.image.preview_url(img_info.full_storage_url, height=_preview_height)
                })
        progress_cb(1)


@g.my_app.callback("show_tags")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def show_tags(api: sly.Api, task_id, context, state, app_logger):
    global tag2images, tag2urls
    tag2images = defaultdict(lambda: defaultdict(list))
    tag2urls = defaultdict(list)

    progress = get_progress_cb(progress_index, "Calculating stats", g.project_info.items_count)
    init_cache(splits.train_set, "train", progress)
    init_cache(splits.val_set, "val", progress)

    segments = [
        {"name": "train", "key": "train", "color": "#13ce66"},
        {"name": "val", "key": "val", "color": "#ffa500"},
        # 20a0ff
    ]

    max_count = -1
    tags_balance_rows = []
    # tags with 0 images will be ignored automatically
    for tag_name, segment_infos in tag2images.items():
        if tag_name.lower() in _ignore_tags:
            continue
        tag_meta = g.project_meta.get_tag_meta(tag_name)
        tag_meta: sly.TagMeta
        if tag_meta.value_type not in _allowed_tag_types:
            continue

        train_count = len(segment_infos["train"])
        val_count = len(segment_infos["val"])

        #@TODO: for debug
        # train_count = random.randint(0, train_count)
        # val_count = random.randint(0, val_count)

        total = train_count + val_count
        tags_balance_rows.append({
            "name": tag_name,
            "total": total,
            "segments": {
                "train": train_count,
                "val": val_count,
            }
        })
        max_count = max(max_count, total)
    reset_progress(progress_index)

    rows_sorted = sorted(tags_balance_rows, key=lambda k: k["total"], reverse=True)
    tags_balance = {
        "maxValue": max_count,
        "segments": segments,
        "rows": rows_sorted
    }

    subsample_urls = {tag_name: urls[:_max_examples_count] for tag_name, urls in tag2urls.items()}

    fields = [
        {"field": "data.done3", "payload": True},
        {"field": "state.tagsInProgress", "payload": False},
    ]
    g.api.app.set_fields(g.task_id, fields)

    fields = [
        {"field": "data.tagsBalance", "payload": tags_balance},
        {"field": "data.tag2urls", "payload": subsample_urls},
    ]
    g.api.app.set_fields(g.task_id, fields)
