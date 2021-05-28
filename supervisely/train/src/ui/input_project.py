import numpy as np
import supervisely_lib as sly
import sly_globals as g


def init(data):
    data["projectId"] = g.project_info.id
    data["projectName"] = g.project_info.name
    data["projectImagesCount"] = g.project_info.items_count
    data["projectPreviewUrl"] = g.api.image.preview_url(g.project_info.reference_image_url, 100, 100)


def clean_bad_images_from_project(project_dir):
    project = sly.Project(project_dir, sly.OpenMode.READ)
    train_tags = sly.json.load_json_file(project_dir, "gt_labels.json")

    for dataset in project.datasets:
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
                dataset.delete_item(item_name)
            if num_training_tags_on_image > 1:
                sly.logger.warn(f"Conflict: multiple training tags were assigned to image {item_name} in dataset {dataset.name}, will be ignored")
                dataset.delete_item(item_name)
