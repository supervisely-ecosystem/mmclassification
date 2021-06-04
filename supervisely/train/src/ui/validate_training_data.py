from collections import defaultdict
import supervisely_lib as sly
import sly_globals as g
import splits
import tags


final_tags = []
final_tags2images = defaultdict(lambda: defaultdict(list))


def init(data, state):
    state["collapsed4"] = True
    state["disabled4"] = True


@g.my_app.callback("validate_data")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def validate_data(api: sly.Api, task_id, context, state, app_logger):
    report = []
    report.append({
        "title": "total images in project",
        "count": g.project_info.items_count,
        "type": "info",
    })
    report.append({
        "title": "total tags in project",
        "count": len(g.project_meta.tag_metas),
        "type": "info",
    })

    training_tags = state["selectedTags"]
    report.append({
        "title": "training tags",
        "count": len(training_tags),
        "type": "info",
    })

    report.append({
        "title": "tags unavailable for training",
        "count": len(tags.disabled_tags),
        "type": "warning",
        "description": "see previous step for more info"
    })

    report.append({
        "title": "images without tags",
        "count": len(tags.images_without_tags),
        "type": "warning",
        "description": "such images does not have any tags, these images will ignored and not be used for training"
    })

    collisions = defaultdict(int)
    for tag_name in training_tags:
        for split, infos in tags.tag2images[tag_name].items():
            for info in infos:
                collisions[info.id] += 1
    num_collision_images = 0
    for image_id, counter in collisions.items():
        if counter > 1:
            num_collision_images += 1
    report.append({
        "title": "images with collisions",
        "count": len(num_collision_images),
        "type": "warning",
        "description": "images have more that one training tags assigned, such images will be ignored and removed from train/val sets"
    })

    # remove collision images from sets
    global final_tags2images, final_tags
    final_images_count = 0
    final_train_size = 0
    final_val_size = 0
    for tag_name in training_tags:
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
    for tag_name in training_tags:
        if len(final_tags2images[tag_name]["train"]) == 0:
            ignore_tags_after_validation.append(tag_name)
        else:
            final_tags.append(tag_name)


    #
    #     {
    #         "type": "info",
    #         "name": "total images",
    #         "count": 8886263,
    #     },
    #     {
    #         "type": "info",
    #         "count": 245,
    #         "name": "total tags",
    #     },
    #     {
    #         "type": "accept",
    #         "count": 245,
    #         "name": "training tags",
    #         "description": "number of selected tags for training"
    #     },
    #     {
    #         "type": "warning",
    #         "count": 0,
    #         "name": "unavailable tags",
    #         "description": "tags that can not be used in training",
    #     },
    #     {
    #         "type": "warning",
    #         "count": 0,
    #         "name": "unavailable tags",
    #         "description": "tags that can not be used in training",
    #     }
    # ]


    total_images = g.project_info.items_count

    final_train_set_size = len(splits.train_set)
    final_val_set_size = len(splits.val_set)
    train_val_intersection = 0
    images_not_in_train_val_splits = 0

    tags_in_project = len(g.project_meta.tag_metas)
    training_tags = len(state["selectedTags"])

    tags_without_training_samples = 0
    tags_without_validation_samples = 0

    images_without_target_tags = 0
    images_with_collisions = 0



