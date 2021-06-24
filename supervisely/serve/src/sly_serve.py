import os

import globals as g
import nn_utils
import supervisely_lib as sly


@g.my_app.callback("get_model_meta")
@sly.timeit
def get_model_meta(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=g.meta.to_json())


@g.my_app.callback("get_tags_examples")
@sly.timeit
def get_tags_examples(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=g.labels_urls)


@g.my_app.callback("get_session_info")
@sly.timeit
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    info = {
        "app": "MM Classification Serve",
        "weights": g.remote_weights_path,
        "device": g.device,
        "session_id": task_id,
        "tags_count": len(g.meta.tag_metas),
    }
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=info)


def inference_image_path(image_path, context, state, app_logger):
    app_logger.debug("Input path", extra={"path": image_path})
    res_path = image_path
    if "rectangle" in state:
        top, left, bottom, right = state["rectangle"]
        rect = sly.Rectangle(top, left, bottom, right)

        image = sly.image.read(image_path)  # RGB image
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

    res = nn_utils.inference_model(g.model, res_path, topn=5)
    return res
    # app_logger.debug("Input path", extra={"path": image_path})
    #
    # rect = None
    # if "rectangle" in state:
    #     top, left, bottom, right = state["rectangle"]
    #     rect = sly.Rectangle(top, left, bottom, right)
    #
    # settings = state.get("settings", {})
    # for key, value in default_settings.items():
    #     if key not in settings:
    #         app_logger.warn("Field {!r} not found in inference settings. Use default value {!r}".format(key, value))
    # debug_visualization = settings.get("debug_visualization", default_settings["debug_visualization"])
    # conf_thres = settings.get("conf_thres", default_settings["conf_thres"])
    # iou_thres = settings.get("iou_thres", default_settings["iou_thres"])
    # augment = settings.get("augment", default_settings["augment"])
    #
    # image = sly.image.read(image_path)  # RGB image
    # if rect is not None:
    #     canvas_rect = sly.Rectangle.from_size(image.shape[:2])
    #     results = rect.crop(canvas_rect)
    #     if len(results) != 1:
    #         return {
    #             "message": "roi rectangle out of image bounds",
    #             "roi": state["rectangle"],
    #             "img_size": {"height": image.shape[0], "width": image.shape[1]}
    #         }
    #     rect = results[0]
    #     image = sly.image.crop(image, rect)
    # ann_json = inference(model, half, device, imgsz, stride, image, meta,
    #                      conf_thres=conf_thres, iou_thres=iou_thres, augment=augment,
    #                      debug_visualization=debug_visualization)
    # return ann_json


@g.my_app.callback("inference_image_url")
@sly.timeit
def inference_image_url(api: sly.Api, task_id, context, state, app_logger):
    pass
    # app_logger.debug("Input data", extra={"state": state})
    #
    # image_url = state["image_url"]
    # ext = sly.fs.get_file_ext(image_url)
    # if ext == "":
    #     ext = ".jpg"
    # local_image_path = os.path.join(my_app.data_dir, sly.rand_str(15) + ext)
    #
    # sly.fs.download(image_url, local_image_path)
    # ann_json = inference_image_path(local_image_path, context, state, app_logger)
    # sly.fs.silent_remove(local_image_path)
    #
    # request_id = context["request_id"]
    # my_app.send_response(request_id, data=ann_json)


@g.my_app.callback("inference_image_id")
@sly.timeit
def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
    pass
    # app_logger.debug("Input data", extra={"state": state})
    # image_id = state["image_id"]
    # image_info = api.image.get_info_by_id(image_id)
    # image_path = os.path.join(my_app.data_dir, sly.rand_str(10) + image_info.name)
    # api.image.download_path(image_id, image_path)
    # ann_json = inference_image_path(image_path, context, state, app_logger)
    # sly.fs.silent_remove(image_path)
    # request_id = context["request_id"]
    # my_app.send_response(request_id, data=ann_json)


@g.my_app.callback("inference_batch_ids")
@sly.timeit
def inference_batch_ids(api: sly.Api, task_id, context, state, app_logger):
    pass
    # app_logger.debug("Input data", extra={"state": state})
    # ids = state["batch_ids"]
    # infos = api.image.get_info_by_id_batch(ids)
    # paths = []
    # for info in infos:
    #     paths.append(os.path.join(my_app.data_dir, sly.rand_str(10) + info.name))
    # api.image.download_paths(infos[0].dataset_id, ids, paths)
    #
    # results = []
    # for image_path in paths:
    #     ann_json = inference_image_path(image_path, context, state, app_logger)
    #     results.append(ann_json)
    #     sly.fs.silent_remove(image_path)
    #
    # request_id = context["request_id"]
    # my_app.send_response(request_id, data=results)


def debug_inference():
    image_id = 903277
    image_path = f"./data/images/{image_id}.jpg"
    if not sly.fs.file_exists(image_path):
        g.my_app.public_api.image.download_path(image_id, image_path)
    res = nn_utils.inference_model(g.model, image_path, topn=5)


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "modal.state.weightsPath": g.remote_weights_path,
        "device": g.device
    })

    nn_utils.download_model_and_configs()
    nn_utils.construct_model_meta()
    nn_utils.deploy_model()

    g.my_app.run()


#@TODO: inference_image_path topn
#@TODO: handle exceptions in every callback and return error back
#@TODO: add select device with groups
#@TODO: release new sdk with api.file.list2
#@TODO: readme + gif - how to replace tag2urls file + release another app
#@TODO: interface to replace tag2urls file
if __name__ == "__main__":
    sly.main_wrapper("main", main)