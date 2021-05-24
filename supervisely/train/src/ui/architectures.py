import errno
import os
import requests
import sly_globals as g
from sly_train_progress import get_progress_cb
import supervisely_lib as sly


def get_models_list():
    return [
        {
            "config": "configs/vgg/vgg11_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_batch256_imagenet_20210208-4271cd6c.pth",
            "model": "VGG-11",
            "params": "132.86",
            "flops": "7.63",
            "top1": "68.75",
            "top5": "88.87"
        },
        {
            "config": "configs/vgg/vgg13_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vgg/vgg13_batch256_imagenet_20210208-4d1d6080.pth",
            "model": "VGG-13",
            "params": "133.05",
            "flops": "11.34",
            "top1": "70.02",
            "top5": "89.46"
        },
        {
            "config": "configs/vgg/vgg16_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_batch256_imagenet_20210208-db26f1a5.pth",
            "model": "VGG-16",
            "params": "138.36",
            "flops": "15.5",
            "top1": "71.62",
            "top5": "90.49"
        },
        {
            "config": "configs/vgg/vgg19_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_bn_batch256_imagenet_20210208-da620c4f.pth",
            "model": "VGG-19",
            "params": "143.67",
            "flops": "19.67",
            "top1": "72.41",
            "top5": "90.80"
        },
        {
            "config": "configs/vgg/vgg11bn_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_bn_batch256_imagenet_20210207-f244902c.pth",
            "model": "VGG-11-BN",
            "params": "132.87",
            "flops": "7.64",
            "top1": "70.75",
            "top5": "90.12"
        },
        {
            "config": "configs/vgg/vgg13bn_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vgg/vgg13_bn_batch256_imagenet_20210207-1a8b7864.pth",
            "model": "VGG-13-BN",
            "params": "133.05",
            "flops": "11.36",
            "top1": "72.15",
            "top5": "90.71"
        },
        {
            "config": "configs/vgg/vgg16_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_bn_batch256_imagenet_20210208-7e55cd29.pth",
            "model": "VGG-16-BN",
            "params": "138.37",
            "flops": "15.53",
            "top1": "73.72",
            "top5": "91.68"
        },
        {
            "config": "configs/vgg/vgg19bn_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_bn_batch256_imagenet_20210208-da620c4f.pth",
            "model": "VGG-19-BN",
            "params": "143.68",
            "flops": "19.7",
            "top1": "74.70",
            "top5": "92.24"
        },
        {
            "config": "configs/resnet/resnet18_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_batch256_imagenet_20200708-34ab8f90.pth",
            "model": "ResNet-18",
            "params": "11.69",
            "flops": "1.82",
            "top1": "70.07",
            "top5": "89.44"
        },
        {
            "config": "configs/resnet/resnet34_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_batch256_imagenet_20200708-32ffb4f7.pth",
            "model": "ResNet-34",
            "params": "21.8",
            "flops": "3.68",
            "top1": "73.85",
            "top5": "91.53"
        },
        {
            "config": "configs/resnet/resnet50_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth",
            "model": "ResNet-50",
            "params": "25.56",
            "flops": "4.12",
            "top1": "76.55",
            "top5": "93.15"
        },
        {
            "config": "configs/resnet/resnet101_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_batch256_imagenet_20200708-753f3608.pth",
            "model": "ResNet-101",
            "params": "44.55",
            "flops": "7.85",
            "top1": "78.18",
            "top5": "94.03"
        },
        {
            "config": "configs/resnet/resnet152_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_batch256_imagenet_20200708-ec25b1f9.pth",
            "model": "ResNet-152",
            "params": "60.19",
            "flops": "11.58",
            "top1": "78.63",
            "top5": "94.16"
        },
        {
            "config": "",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnest/resnest50_imagenet_converted-1ebf0afe.pth",
            "model": "ResNeSt-50*",
            "params": "27.48",
            "flops": "5.41",
            "top1": "81.13",
            "top5": "95.59"
        },
        {
            "config": "",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnest/resnest101_imagenet_converted-032caa52.pth",
            "model": "ResNeSt-101*",
            "params": "48.28",
            "flops": "10.27",
            "top1": "82.32",
            "top5": "96.24"
        },
        {
            "config": "",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnest/resnest200_imagenet_converted-581a60f2.pth",
            "model": "ResNeSt-200*",
            "params": "70.2",
            "flops": "17.53",
            "top1": "82.41",
            "top5": "96.22"
        },
        {
            "config": "",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnest/resnest269_imagenet_converted-59930960.pth",
            "model": "ResNeSt-269*",
            "params": "110.93",
            "flops": "22.58",
            "top1": "82.70",
            "top5": "96.28"
        },
        {
            "config": "configs/resnet/resnetv1d50_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d50_batch256_imagenet_20200708-1ad0ce94.pth",
            "model": "ResNetV1D-50",
            "params": "25.58",
            "flops": "4.36",
            "top1": "77.4",
            "top5": "93.66"
        },
        {
            "config": "configs/resnet/resnetv1d101_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d101_batch256_imagenet_20200708-9cb302ef.pth",
            "model": "ResNetV1D-101",
            "params": "44.57",
            "flops": "8.09",
            "top1": "78.85",
            "top5": "94.38"
        },
        {
            "config": "configs/resnet/resnetv1d152_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d152_batch256_imagenet_20200708-e79cb6a2.pth",
            "model": "ResNetV1D-152",
            "params": "60.21",
            "flops": "11.82",
            "top1": "79.35",
            "top5": "94.61"
        },
        {
            "config": "configs/resnext/resnext50_32x4d_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnext/resnext50_32x4d_batch256_imagenet_20200708-c07adbb7.pth",
            "model": "ResNeXt-32x4d-50",
            "params": "25.03",
            "flops": "4.27",
            "top1": "77.92",
            "top5": "93.74"
        },
        {
            "config": "configs/resnext/resnext101_32x4d_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x4d_batch256_imagenet_20200708-87f2d1c9.pth",
            "model": "ResNeXt-32x4d-101",
            "params": "44.18",
            "flops": "8.03",
            "top1": "78.7",
            "top5": "94.34"
        },
        {
            "config": "configs/resnext/resnext101_32x8d_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x8d_batch256_imagenet_20200708-1ec34aa7.pth",
            "model": "ResNeXt-32x8d-101",
            "params": "88.79",
            "flops": "16.5",
            "top1": "79.22",
            "top5": "94.52"
        },
        {
            "config": "configs/resnext/resnext152_32x4d_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/resnext/resnext152_32x4d_batch256_imagenet_20200708-aab5034c.pth",
            "model": "ResNeXt-32x4d-152",
            "params": "59.95",
            "flops": "11.8",
            "top1": "79.06",
            "top5": "94.47"
        },
        {
            "config": "configs/seresnet/seresnet50_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/se-resnet/se-resnet50_batch256_imagenet_20200804-ae206104.pth",
            "model": "SE-ResNet-50",
            "params": "28.09",
            "flops": "4.13",
            "top1": "77.74",
            "top5": "93.84"
        },
        {
            "config": "configs/seresnet/seresnet101_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/se-resnet/se-resnet101_batch256_imagenet_20200804-ba5b51d4.pth",
            "model": "SE-ResNet-101",
            "params": "49.33",
            "flops": "7.86",
            "top1": "78.26",
            "top5": "94.07"
        },
        {
            "config": "configs/shufflenet_v1/shufflenet_v1_1x_b64x16_linearlr_bn_nowd_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/shufflenet_v1/shufflenet_v1_batch1024_imagenet_20200804-5d6cec73.pth",
            "model": "ShuffleNetV1 1.0x (group=3)",
            "params": "1.87",
            "flops": "0.146",
            "top1": "68.13",
            "top5": "87.81"
        },
        {
            "config": "configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/shufflenet_v2/shufflenet_v2_batch1024_imagenet_20200812-5bf4721e.pth",
            "model": "ShuffleNetV2 1.0x",
            "params": "2.28",
            "flops": "0.149",
            "top1": "69.55",
            "top5": "88.92"
        },
        {
            "config": "configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth",
            "model": "MobileNet V2",
            "params": "3.5",
            "flops": "0.319",
            "top1": "71.86",
            "top5": "90.42"
        },
        {
            "config": "configs/vision_transformer/vit_base_patch16_384_finetune_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vit/vit_base_patch16_384.pth",
            "model": "ViT-B/16*",
            "params": "86.86",
            "flops": "33.03",
            "top1": "84.20",
            "top5": "97.18"
        },
        {
            "config": "configs/vision_transformer/vit_base_patch32_384_finetune_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vit/vit_base_patch32_384.pth",
            "model": "ViT-B/32*",
            "params": "88.3",
            "flops": "8.56",
            "top1": "81.73",
            "top5": "96.13"
        },
        {
            "config": "configs/vision_transformer/vit_large_patch16_384_finetune_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vit/vit_large_patch16_384.pth",
            "model": "ViT-L/16*",
            "params": "304.72",
            "flops": "116.68",
            "top1": "85.08",
            "top5": "97.38"
        },
        {
            "config": "configs/vision_transformer/vit_large_patch32_384_finetune_imagenet.py",
            "weightsUrl": "https://download.openmmlab.com/mmclassification/v0/vit/vit_large_patch32_384.pth",
            "model": "ViT-L/32*",
            "params": "306.63",
            "flops": "29.66",
            "top1": "81.52",
            "top5": "96.06"
        }
    ]


def get_table_columns():
    return [
        {"key": "model", "title": "Model", "subtitle": None},
        {"key": "params", "title": "Params (M)", "subtitle": None},
        {"key": "flops", "title": "Flops (G)", "subtitle": None},
        {"key": "top1", "title": "Top-1 (%)", "subtitle": None},
        {"key": "top5", "title": "Top-5 (%)", "subtitle": None},
    ]


def get_model_info_by_name(name):
    models = get_models_list()
    for info in models:
        if info["model"] == name:
            return info
    raise KeyError(f"Model {name} not found")


def get_pretrained_weights_by_name(name):
    return get_model_info_by_name(name)["weightsUrl"]


def init(data, state):
    data["models"] = get_models_list()
    data["modelColumns"] = get_table_columns()
    state["selectedModel"] = "ResNet-34"
    state["weightsInitialization"] = "imagenet"

    # @TODO: for debug
    # state["weightsPath"] = "/yolov5_train/coco128_002/2390/weights/best.pt"
    state["weightsPath"] = ""


def prepare_weights(state):
    if state["weightsInitialization"] == "custom":
        # download custom weights
        weights_path_remote = state["weightsPath"]
        if not weights_path_remote.endswith(".pt"):
            raise ValueError(f"Weights file has unsupported extension {sly.fs.get_file_ext(weights_path_remote)}. "
                             f"Supported: '.pt'")
        weights_path_local = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_path_remote))
        file_info = g.api.file.get_info_by_path(g.team_id, weights_path_remote)
        if file_info is None:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), weights_path_remote)
        progress_cb = get_progress_cb("Download weights", file_info.sizeb, is_size=True)
        g.api.file.download(g.team_id, weights_path_remote, weights_path_local, g.my_app.cache, progress_cb)

        state["_weightsPath"] = weights_path_remote
        state["weightsPath"] = weights_path_local
    else:
        weights_url = get_pretrained_weights_by_name(state["selectedModel"])
        weights_path_local = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_url))
        if sly.fs.file_exists(weights_path_local) is False: # speedup for debug, has no effects in production
            response = requests.head(weights_url, allow_redirects=True)
            sizeb = int(response.headers.get('content-length', 0))
            progress_cb = get_progress_cb("Download weights", sizeb, is_size=True)
            sly.fs.download(weights_url, weights_path_local, g.my_app.cache, progress_cb)

        state["weightsPath"] = weights_path_local
        sly.logger.info("Pretrained ImageNet weights has been successfully downloaded")
