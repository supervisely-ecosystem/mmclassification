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
_max_examples_count = 20

_ignore_tags = ["train", "val"]
_allowed_tag_types = [sly.TagValueType.NONE]

image_slider_options = {
    "selectable": False,
    "height": f"{_preview_height}px"
}

# speedup during debug (has no effects in production)
# cache_base_filename = os.path.join(g.my_app.data_dir, f"{g.project_id}")
# cache_path = cache_base_filename + ".db"


def init(data, state):
    data["tagsBalance"] = None
    state["selectedTags"] = []
    state["tagsInProgress"] = False
    data["tagsBalanceOptions"] = {
        "selectable": True,
        "collapsable": True,
        "clickableName": False,
        "clickableSegment": False,
        "maxHeight": "400px"
    }
    data["imageSliderOptions"] = image_slider_options
    data["done3"] = False
    init_progress(progress_index, data)


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

    progress = get_progress_cb(progress_index, "Calculate stats", g.project_info.items_count)
    init_cache(splits.train_set, "train", progress)
    init_cache(splits.val_set, "val", progress)

    segments = [
        {"name": "train", "key": "train", "color": "#13ce66"},
        {"name": "val", "key": "val", "color": "#ffa500"},
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

        # @TODO: for debug
        train_count = random.randint(0, train_count)
        val_count = random.randint(0, val_count)

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
