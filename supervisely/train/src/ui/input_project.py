import numpy as np
import supervisely_lib as sly
import sly_globals as g


def init(data):
    data["projectId"] = g.project_info.id
    data["projectName"] = g.project_info.name
    data["projectImagesCount"] = g.project_info.items_count
    data["projectPreviewUrl"] = g.api.image.preview_url(g.project_info.reference_image_url, 100, 100)


def clean_bad_images_from_project(project_dir, train_set, val_set):
    project = sly.Project(project_dir, sly.OpenMode.READ)
    train_tags = sly.json.load_json_file(project_dir, "gt_labels.json")

    to_remove = {}
    for dataset in project.datasets:
        to_remove[dataset.name] = {}
        for item_name in dataset:
            img_path, ann_path = dataset.get_item_paths(item_name)
            ann = sly.Annotation.load_json_file(ann_path, project.meta)

            num_training_tags_on_image = 0
            for tag in ann.img_tags:
                tag: sly.Tag
                if tag.name in train_tags:
                    num_training_tags_on_image += 1

            if num_training_tags_on_image == 0:
                sly.logger.warn(f"Image {item_name} in dataset {dataset.name} does not have any any of the training tags, will be ignored")
                to_remove[dataset.name][item_name] = 1
                dataset.delete_item(item_name)  # to be sure that there are no bad images in training
            if num_training_tags_on_image > 1:
                sly.logger.warn(f"Conflict: multiple training tags were assigned to image {item_name} in dataset {dataset.name}, will be ignored")
                to_remove[dataset.name][item_name] = 1
                dataset.delete_item(item_name)  # to be sure that there are no bad images in training

    def _filter(items, to_remove):
        filtered = []
        for item in items:
            if item.name in to_remove[item.dataset_name]:
                continue  # remove item
            filtered.append(item)
        return filtered

    filtered_train_set = _filter(train_set, to_remove)
    filtered_val_set = _filter(val_set, to_remove)
    return filtered_train_set, filtered_val_set