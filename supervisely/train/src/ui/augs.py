import supervisely_lib as sly


def get_aug_templates_list():
    return [
        {
            "config": "configs/vgg/vgg11_b32x8_imagenet.py",
            "name": "VGG-11",
            "color": "light",
            "blur": True,
            "noise": True,
            "cutout": True,
            "rotate": True,
            "geometric": 123,
            "fliplr": False,
            "flipud": False,
        },
    ]


def init(data, state):
    state["augsType"] = "template"
