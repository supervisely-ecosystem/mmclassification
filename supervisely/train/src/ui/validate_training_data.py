from collections import defaultdict
import supervisely_lib as sly
import sly_globals as g
import splits
import tags

report = []
final_tags = []
final_tags2images = defaultdict(lambda: defaultdict(list))


def init(data, state):
    state["collapsed4"] = True
    state["disabled4"] = True
    data["validationReport"] = None


@g.my_app.callback("validate_data")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def validate_data(api: sly.Api, task_id, context, state, app_logger):
    report.append({
        "type": "info",
        "title": "Total tags in project",
        "count": len(g.project_meta.tag_metas),
        "description": None
    })

    report.append({
        "title": "Tags unavailable for training",
        "count": len(tags.disabled_tags),
        "type": "warning",
        "description": "See previous step for more info"
    })

    selected_tags = tags.selected_tags  # state["selectedTags"]
    report.append({
        "title": "Selected tags for training",
        "count": len(selected_tags),
        "type": "info",
        "description": None
    })

    report.append({
        "type": "info",
        "title": "Total images in project",
        "count": g.project_info.items_count,
    })

    report.append({
        "title": "Images without tags",
        "count": len(tags.images_without_tags),
        "type": "warning" if len(tags.images_without_tags) > 0 else "pass",
        "description": "Such images don't have any tags so they will ignored and will not be used for training. "
    })

    num_images_before_validation = 0
    for tag_name in selected_tags:
        for split, infos in tags.tag2images[tag_name].items():
            num_images_before_validation += len(infos)
    report.append({
        "title": "Images with training tags",
        "count": num_images_before_validation,
        "type": "error" if num_images_before_validation == 0 else "pass",
        "description": "Images that have one of the selected tags assigned (before validation)"
    })

    collisions = defaultdict(int)
    for tag_name in selected_tags:
        for split, infos in tags.tag2images[tag_name].items():
            for info in infos:
                collisions[info.id] += 1
    num_collision_images = 0
    for image_id, counter in collisions.items():
        if counter > 1:
            num_collision_images += 1
    report.append({
        "title": "Images with tags collisions",
        "count": num_collision_images,
        "type": "warning" if num_collision_images > 0 else "pass",
        "description": "images with more than one training tags assigned, they will be removed from train/val sets"
    })

    # remove collision images from sets
    final_images_count = 0
    final_train_size = 0
    final_val_size = 0
    for tag_name in selected_tags:
        for split, infos in tags.tag2images[tag_name].items():
            for info in infos:
                if collisions[info.id] == 0:
                    final_tags2images[tag_name][split].append(info)
                    final_images_count += 1
                    if split == "train":
                        final_train_size += 1
                    else:
                        final_val_size += 1
        if tag_name in final_tags2images and len(final_tags2images[tag_name]["train"]) > 0:
            final_tags.append(tag_name)

    report.append({
        "title": "Train set size",
        "count": final_train_size,
        "type": "error" if final_train_size == 0 else "pass",
        "description": "Size of training set after removing images with collisions"
    })
    report.append({
        "title": "Val set size",
        "count": final_val_size,
        "type": "error" if final_val_size == 0 else "pass",
        "description": "Size of validation set after removing images with collisions"
    })
    report.append({
        "title": "Final training tags",
        "count": len(final_tags),
        "type": "error" if len(final_tags) < 2 else "pass",
        "description": "If this number differs from the number of selected tags then it means that after data validation and "
                       "cleaning some of the selected tags have 0 examples in train set"
    })

    report.append({
        "title": "images with training tags",
        "count": len(tags.images_without_tags),
        "type": "pass",
        "description": "one of the selected training tags is assigned to these images"
    })

    # remove collision images from sets
    final_images_count = 0
    final_train_size = 0
    final_val_size = 0
    for tag_name in selected_tags:
        for split, infos in tags.tag2images[tag_name].items():
            for info in infos:
                if collisions[info.id] :
                    final_tags2images[tag_name][split].append(info)
                    final_images_count += 1
                    if split == "train":
                        final_train_size += 1
                    else:
                        final_val_size += 1

    ignore_tags_after_validation = []
    for tag_name in selected_tags:
        if len(final_tags2images[tag_name]["train"]) == 0:
            ignore_tags_after_validation.append(tag_name)
        else:
            final_tags.append(tag_name)

    fields = [
        {"field": "data.validationReport", "payload": report},
    ]
    g.api.app.set_fields(g.task_id, fields)


