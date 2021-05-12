import supervisely_lib as sly


def init(data, state, project_meta: sly.ProjectMeta):
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