import cv2
import numpy as np
import supervisely as sly

from supervisely.sly_logger import logger

import globals as g


def get_images_ids_to_indexes_mapping(images_ids):
    imagesids2indexes = {}

    for index, image_id in enumerate(images_ids):
        imagesids2indexes.setdefault(image_id, []).append(index)

    return imagesids2indexes


def get_nps_images(images_ids):
    uniqueids2indexes = get_images_ids_to_indexes_mapping(images_ids)

    unique_images_ids = list(uniqueids2indexes.keys())

    images_infos = g.api.image.get_info_by_id_batch(unique_images_ids)
    images_ids = np.asarray(images_ids)

    dataset2ids = {}
    for index, image_info in enumerate(images_infos):  # group images by datasets
        dataset2ids.setdefault(image_info.dataset_id, []).append(image_info.id)

    images_nps = [_ for _ in range(len(images_ids))]  # back to plain

    for ds_id, ids_batch in dataset2ids.items():
        nps_for_ds = g.api.image.download_nps(dataset_id=ds_id, ids=ids_batch)

        for index, image_id in enumerate(ids_batch):
            for image_index in uniqueids2indexes[image_id]:
                images_nps[image_index] = cv2.cvtColor(nps_for_ds[index], cv2.COLOR_BGR2RGB)

    return np.asarray(images_nps)


def crop_images(images_nps, rectangles):
    if rectangles is None:
        return images_nps

    elif len(rectangles) != len(images_nps):
        logger.error(f'{len(rectangles)=} != {len(images_nps)=}')
        raise ValueError(f'{len(rectangles)=} != {len(images_nps)=}')

    cropped_images = []
    for img_np, rectangle in zip(images_nps, rectangles):
        try:
            top, left, bottom, right = rectangle
            rect = sly.Rectangle(top, left, bottom, right)
            cropping_rect = rect.crop(sly.Rectangle.from_size(img_np.shape[:2]))[0]
            cropped_image = sly.image.crop(img_np, cropping_rect)
            cropped_images.append(cropped_image)
        except Exception as ex:
            cropped_images.append(None)
            logger.warning(f'Cannot crop image: {ex}')

    return np.asarray(cropped_images)

