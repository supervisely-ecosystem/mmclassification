from collections import defaultdict
import supervisely_lib as sly
import sly_globals as g

tag2images = defaultdict(list)
tag2urls = defaultdict(list)

_preview_height = 200


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
    #cache_images_examples(data)


def cache_images_examples(data):
    pass
    # progress = sly.Progress("Caching image examples for tags", g.api.project.get_images_count(g.project_id))
    # for dataset in g.api.dataset.get_list(g.project_id):
    #     ds_images = g.api.image.get_list(dataset.id)
    #     for batch in sly.batched(ds_images):
    #         ids = [info.id for info in ds_images]
    #         anns_infos = g.api.annotation.download_batch(dataset.id, ids)
    #         anns_jsons = [ann_info.annotation for ann_info in anns_infos]
    #         anns = [sly.Annotation.from_json(ann_json, g.project_meta) for ann_json in anns_jsons]
    #         for ann, img_info in zip(anns, batch):
    #             for tag in ann.img_tags:
    #                 tag: sly.Tag
    #                 if len(tag2urls[tag.name]) >= 5:
    #                     continue
    #                 tag2images[tag.name].append(img_info)
    #                 tag2urls[tag.name].append({
    #                     "fullSize": img_info.full_storage_url,
    #                     "preview": g.api.image.preview_url(img_info.full_storage_url, height=_preview_height)
    #                 })
    #         progress.iters_done_report(len(batch))
    #     break  #@TODO: for debug
    # data["tag2urls"] = tag2urls
    # data["tag2urls"] = {
    #     "1000591": [
    #         {
    #             "moreExamples": [
    #                 "https://www.w3schools.com/howto/img_nature.jpg",
    #                 "https://www.w3schools.com/howto/img_nature.jpg",
    #                 "https://www.w3schools.com/howto/img_nature.jpg",
    #             ],
    #             "preview": "https://www.w3schools.com/howto/img_nature.jpg"
    #         },
    #         {
    #             "fullSize": "https://www.quackit.com/pix/samples/18m.jpg",
    #             "preview": "https://www.quackit.com/pix/samples/18m.jpg"
    #         },
    #         {
    #             "fullSize": "https://images.unsplash.com/photo-1616164942267-446356ac3a34?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=668&q=80",
    #             "preview": "https://images.unsplash.com/photo-1616164942267-446356ac3a34?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=668&q=80"
    #         },
    #         {
    #             "fullSize": "https://images.unsplash.com/photo-1609911569155-c6b221cec943?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=668&q=80",
    #             "preview": "https://images.unsplash.com/photo-1609911569155-c6b221cec943?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=668&q=80"
    #         },
    #         {
    #             "fullSize": "https://images.unsplash.com/photo-1569000972143-d9f60420a1b4?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1500&q=80",
    #             "preview": "https://images.unsplash.com/photo-1569000972143-d9f60420a1b4?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1500&q=80"
    #         }
    #     ],
    # }
