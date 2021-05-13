from collections import defaultdict
import supervisely_lib as sly
import sly_globals as g

tag2images = defaultdict(list)
tag2urls = defaultdict(list)

_preview_height = 100
_max_examples_count = 5


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


def cache_images_examples(data):
    id_to_tagmeta = g.project_meta.tag_metas.get_id_mapping()
    progress = sly.Progress("Caching image examples for tags", g.api.project.get_images_count(g.project_id))
    for dataset in g.api.dataset.get_list(g.project_id):
        ds_images = g.api.image.get_list(dataset.id)
        for img_info in ds_images:
            tags = sly.TagCollection.from_api_response(img_info.tags, g.project_meta.tag_metas, id_to_tagmeta)
            for tag in tags:
                if len(tag2urls[tag.name]) >= _max_examples_count:
                    continue
                tag2images[tag.name].append(img_info)
                tag2urls[tag.name].append({
                    "moreExamples": [img_info.full_storage_url],
                    "preview": g.api.image.preview_url(img_info.full_storage_url, height=_preview_height)
                })
            progress.iter_done_report()
    data["tag2urls"] = tag2urls
    data["tag2images"] = tag2images

