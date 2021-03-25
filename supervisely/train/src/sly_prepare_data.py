import os
import yaml
import supervisely_lib as sly


def transform_label(class_names, img_size, label: sly.Label):
    class_number = class_names.index(label.obj_class.name)
    rect_geometry = label.geometry.to_bbox()
    center = rect_geometry.center
    x_center = round(center.col / img_size[1], 6)
    y_center = round(center.row / img_size[0], 6)
    width = round(rect_geometry.width / img_size[1], 6)
    height = round(rect_geometry.height / img_size[0], 6)
    result = '{} {} {} {} {}'.format(class_number, x_center, y_center, width, height)
    return result


def _create_data_config(output_dir, meta: sly.ProjectMeta, keep_classes):
    class_names = []
    class_colors = []
    for obj_class in meta.obj_classes:
        if obj_class.name in keep_classes:
            class_names.append(obj_class.name)
            class_colors.append(obj_class.color)

    data_yaml = {
        "train": os.path.join(output_dir, "images/train"),
        "val": os.path.join(output_dir, "images/val"),
        "labels_train": os.path.join(output_dir, "labels/train"),
        "labels_val": os.path.join(output_dir, "labels/val"),
        "nc": len(class_names),
        "names": class_names,
        "colors": class_colors
    }
    sly.fs.mkdir(data_yaml["train"])
    sly.fs.mkdir(data_yaml["val"])
    sly.fs.mkdir(data_yaml["labels_train"])
    sly.fs.mkdir(data_yaml["labels_val"])

    config_path = os.path.join(output_dir, 'data_config.yaml')
    with open(config_path, 'w') as f:
        _ = yaml.dump(data_yaml, f, default_flow_style=None)

    return data_yaml


def transform_annotation(ann, class_names, save_path):
    yolov5_ann = []
    for label in ann.labels:
        if label.obj_class.name in class_names:
            yolov5_ann.append(transform_label(class_names, ann.img_size, label))

    with open(save_path, 'w') as file:
        file.write("\n".join(yolov5_ann))

    if len(yolov5_ann) == 0:
        return True
    return False


def _process_split(project, class_names, images_dir, labels_dir, split, progress_cb):
    for batch in sly.batched(split, batch_size=max(int(len(split) / 50), 10)):
        for dataset_name, item_name in batch:
            dataset = project.datasets.get(dataset_name)
            ann_path = dataset.get_ann_path(item_name)
            ann_json = sly.json.load_json_file(ann_path)
            ann = sly.Annotation.from_json(ann_json, project.meta)

            save_ann_path = os.path.join(labels_dir, f"{sly.fs.get_file_name(item_name)}.txt")
            empty = transform_annotation(ann, class_names, save_ann_path)
            if empty:
                sly.logger.warning(f"Empty annotation dataset={dataset_name} image={item_name}")

            img_path = dataset.get_img_path(item_name)
            save_img_path = os.path.join(images_dir, item_name)
            sly.fs.copy_file(img_path, save_img_path)

        progress_cb(len(batch))


def filter_and_transform_labels(input_dir, train_classes,
                                train_split, val_split,
                                output_dir, progress_cb):
    project = sly.Project(input_dir, sly.OpenMode.READ)
    data_yaml = _create_data_config(output_dir, project.meta, train_classes)

    _process_split(project, data_yaml["names"], data_yaml["train"], data_yaml["labels_train"], train_split, progress_cb)
    _process_split(project, data_yaml["names"], data_yaml["val"], data_yaml["labels_val"], val_split, progress_cb)

