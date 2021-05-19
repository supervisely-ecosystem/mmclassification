import os
from collections import defaultdict, namedtuple
import shelve
import supervisely_lib as sly
import sly_globals as g
import random

tag2images = defaultdict(list)
#all_images = []
tag2urls = defaultdict(list)

_preview_height = 150
_max_examples_count = 9

image_slider_options = {
    "selectable": False,
    "height": f"{_preview_height}px"
}


#speedup during debug (has no effects in production)
cache_base_filename = os.path.join(g.my_app.data_dir, f"{g.project_id}")
cache_path = cache_base_filename + ".db"


def init(data, state):
    stats = g.api.project.get_stats(g.project_id)
    images_with_tag = {}
    for item in stats["imageTags"]["items"]:
        images_with_tag[item["tagMeta"]["name"]] = item["total"]

    tags_json = []
    for tag_meta in g.project_meta.tag_metas.to_json():
        tag_meta["imagesCount"] = images_with_tag[tag_meta["name"]]
        tag_meta["valueType"] = tag_meta["value_type"]
        if tag_meta["valueType"] != sly.TagValueType.NONE or tag_meta["name"] in ["train", "val"]:
            continue
        tags_json.append(tag_meta)
    data["tags"] = tags_json
    state["selectedTags"] = []
    cache_images_examples(data)
    data["imageSliderOptions"] = image_slider_options


def cache_images_examples(data):
    global tag2urls, tag2images

    if sly.fs.file_exists(cache_path):
        sly.logger.info("Cache exists, read tags and images info from cache")
        with shelve.open(cache_base_filename, flag='r') as s:
            tag2urls = s["tag2urls"]
            tag2images = s["tag2images"]
    else:
        id_to_tagmeta = g.project_meta.tag_metas.get_id_mapping()
        progress = sly.Progress("Caching image examples for tags", g.api.project.get_images_count(g.project_id))
        for dataset in g.api.dataset.get_list(g.project_id):
            ds_images = g.api.image.get_list(dataset.id)
            for img_info in ds_images:
                tags = sly.TagCollection.from_api_response(img_info.tags, g.project_meta.tag_metas, id_to_tagmeta)
                for tag in tags:
                    img_info_dict = img_info._asdict()
                    tag2images[tag.name].append(img_info_dict)

                    if len(tag2urls[tag.name]) >= _max_examples_count:
                        continue
                    #all_images.append(img_info)
                    tag2urls[tag.name].append({
                        "moreExamples": [img_info.full_storage_url],
                        "preview": g.api.image.preview_url(img_info.full_storage_url, height=_preview_height)
                    })
                progress.iter_done_report()

        with shelve.open(cache_base_filename) as s:
            s["tag2urls"] = tag2urls
            s["tag2images"] = tag2images
        sly.logger.info(f"Cache for project id={g.project_id} has been successfully created")

    data["tag2urls"] = tag2urls
    #data["tag2images"] = tag2images


def get_random_image():
    rand_key = random.choice(list(tag2images.keys()))
    image_info_dict = random.choice(tag2images[rand_key])
    ImageInfo = namedtuple('ImageInfo', image_info_dict)
    info = ImageInfo(**image_info_dict)
    return info