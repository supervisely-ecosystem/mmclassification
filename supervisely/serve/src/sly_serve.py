import os
import functools
from functools import lru_cache

import cv2
import numpy as np
import supervisely as sly

import globals as g
import functions as f
import nn_utils


@lru_cache(maxsize=10)
def get_image_by_id(image_id):
    img = g.api.image.download_np(image_id)
    return img


def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            request_id = kwargs["context"]["request_id"]
            g.my_app.send_response(request_id, data={"error": repr(e)})
        return value

    return wrapper


@g.my_app.callback("get_model_meta")
@sly.timeit
@send_error_data
def get_model_meta(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=g.meta.to_json())


@g.my_app.callback("get_tags_examples")
@sly.timeit
@send_error_data
def get_tags_examples(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=g.labels_urls)


@g.my_app.callback("get_session_info")
@sly.timeit
@send_error_data
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    info = {
        "app": "MM Classification Serve",
        "weights": g.remote_weights_path,
        "device": g.device,
        "session_id": task_id,
        "classes_count": len(g.meta.tag_metas),
        "classification_mode": g.cls_mode
    }
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=info)


def inference_image_path(image_path, context, state, app_logger):
    app_logger.debug("Input path", extra={"path": image_path})
    res_path = image_path
    if "rectangle" in state:
        image = sly.image.read(image_path)  # RGB image
        top, left, bottom, right = f.get_bbox_with_padding(rectangle=state['rectangle'], pad_percent=state.get('pad', 0),
                                                           img_size=image.shape[:2])  # img_size=(h,w)

        rect = sly.Rectangle(top, left, bottom, right)
        canvas_rect = sly.Rectangle.from_size(image.shape[:2])
        results = rect.crop(canvas_rect)
        if len(results) != 1:
            return {
                "message": "roi rectangle out of image bounds",
                "roi": state["rectangle"],
                "img_size": {"height": image.shape[0], "width": image.shape[1]}
            }
        rect = results[0]
        cropped_image = sly.image.crop(image, rect)
        res_path = os.path.join(g.my_app.data_dir, sly.rand_str(10) + sly.fs.get_file_ext(image_path))
        sly.image.write(res_path, cropped_image)

    res = nn_utils.inference_model(g.model, res_path, topn=state.get("topn", 5))
    if "rectangle" in state:
        sly.fs.silent_remove(res_path)

    return res


@g.my_app.callback("inference_image_url")
@sly.timeit
@send_error_data
def inference_image_url(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_url = state["image_url"]
    ext = sly.fs.get_file_ext(image_url)
    if ext == "":
        ext = ".jpg"
    local_image_path = os.path.join(g.my_app.data_dir, sly.rand_str(15) + ext)
    sly.fs.download(image_url, local_image_path)
    results = inference_image_path(local_image_path, context, state, app_logger)
    sly.fs.silent_remove(local_image_path)

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=results)


@g.my_app.callback("inference_image_id")
@sly.timeit
@send_error_data
def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
    sly.logger.info("infer image id", extra={"state": state})

    image_id = state["image_id"]

    image_info = api.image.get_info_by_id(image_id)
    image_path = os.path.join(g.my_app.data_dir, f"{image_id}{sly.fs.get_file_ext(image_info.name)}")
    img = get_image_by_id(image_id)
    sly.image.write(image_path, img)

    predictions = inference_image_path(image_path, context, state, app_logger)
    sly.fs.silent_remove(image_path)
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=predictions)


@g.my_app.callback("inference_batch_ids")
@sly.timeit
@send_error_data
def inference_batch_ids(api: sly.Api, task_id, context, state, app_logger):
    sly.logger.info("inference batch ids called:", extra={"state": state})

    # load images
    images_nps: np.array = f.get_nps_images(images_ids=state["images_ids"])
    images_to_process: np.array = f.crop_images(images_nps=images_nps,
                                                rectangles=state.get('rectangles'),
                                                padding=state.get('pad', 0))

    # inference images
    images_indexes_to_process = np.asarray([index for index, img_np in enumerate(images_to_process)
                                            if img_np is not None])
    inference_results = nn_utils.inference_model_batch(model=g.model,
                                                       images_nps=images_to_process[images_indexes_to_process],
                                                       topn=state.get('topn', 5))

    # return output
    results = [None for _ in images_nps]
    for index, row in enumerate(inference_results):
        results[images_indexes_to_process[index]] = row

    g.my_app.send_response(request_id=context["request_id"], data=results)


# def debug_inference1():
#     image_id = 927270
#     image_path = f"./data/images/{image_id}.jpg"
#     if not sly.fs.file_exists(image_path):
#         g.my_app.public_api.image.download_path(image_id, image_path)
#     res = nn_utils.inference_model(g.model, image_path, topn=5)
# #
# #
# def debug_inference2():
#     image_id = 927270
#     img_np = cv2.cvtColor(g.my_app.public_api.image.download_np(image_id), cv2.COLOR_BGR2RGB)
#     res = nn_utils.inference_model(g.model, img_np, topn=5)
# #
# #
# def debug_inference3():
#     image_id = 927270
#     img_np = cv2.cvtColor(g.my_app.public_api.image.download_np(image_id), cv2.COLOR_BGR2RGB)
#     res = nn_utils.inference_model_batch(g.model, [img_np, img_np, img_np], topn=5)
#

def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "modal.state.slyFile": g.remote_weights_path,
        "device": g.device
    })

    nn_utils.download_model_and_configs()
    nn_utils.construct_model_meta()
    nn_utils.deploy_model()

    # debug_inference1()
    # debug_inference2()
    # debug_inference3()

    sly.logger.info("nps will be converted to RGB")
    g.my_app.run()


# @TODO: readme + gif - how to replace tag2urls file + release another app
if __name__ == "__main__":
    sly.main_wrapper("main", main)
