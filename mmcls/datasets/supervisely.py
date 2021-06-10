import os

import numpy as np
import supervisely_lib as sly

from .base_dataset import BaseDataset
from .builder import DATASETS


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_folders(root):
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        folder_to_idx (dict): the map from folder name to class idx
    """
    folders = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


def get_samples(root, folder_to_idx, extensions):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        extensions (tuple): allowed extensions

    Returns:
        samples (list): a list of tuple where each element is (image, label)
    """
    samples = []
    root = os.path.expanduser(root)
    for folder_name in sorted(os.listdir(root)):
        _dir = os.path.join(root, folder_name)
        if not os.path.isdir(_dir):
            continue

        for _, _, fns in sorted(os.walk(_dir)):
            for fn in sorted(fns):
                if has_file_allowed_extension(fn, extensions):
                    path = os.path.join(folder_name, fn)
                    item = (path, folder_to_idx[folder_name])
                    samples.append(item)
    return samples


@DATASETS.register_module()
class Supervisely(BaseDataset):
    """`Supervisely <https://supervise.ly/>`_ Dataset.
    """

    CLASSES = None

    def __init__(self, project_dir, data_prefix, pipeline, test_mode=False):
        self.gt_labels = sly.json.load_json_file(os.path.join(project_dir, "gt_labels.json"))
        Supervisely.CLASSES = sorted(self.gt_labels, key=self.gt_labels.get)
        self.split_name = data_prefix
        self.items = sly.json.load_json_file(os.path.join(project_dir, "splits.json"))[self.split_name]
        self.project_fs = sly.Project(project_dir, sly.OpenMode.READ)
        super(Supervisely, self).__init__(data_prefix=self.split_name, pipeline=pipeline, test_mode=test_mode)

    def load_annotations(self):
        classes_set = set(Supervisely.CLASSES)
        data_infos = []
        for paths in self.items:
            img_path = paths["img_path"]
            ann_path = paths["ann_path"]
            if not sly.fs.file_exists(img_path):
                sly.logger.warn(f"File {img_path} not found and item will be skipped")
                continue
            if not sly.fs.file_exists(ann_path):
                sly.logger.warn(f"File {ann_path} not found and item will be skipped")
                continue
            ann = sly.Annotation.load_json_file(ann_path, self.project_fs.meta)
            img_tags = {tag.name for tag in ann.img_tags}

            gt_label = list(img_tags.intersection(classes_set))
            if len(gt_label) != 1:
                sly.logger.warn(f"File {ann_path} has {len(gt_label)} gt labels")
                continue
            else:
                gt_label = gt_label[0]

            gt_index = self.gt_labels[gt_label]
            data_infos.append({
                "img_prefix": self.split_name,
                "img_info": {'filename': img_path},
                "gt_label": np.array(gt_index, dtype=np.int64)
            })
        return data_infos
