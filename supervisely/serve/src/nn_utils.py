import os
import pathlib
import supervisely_lib as sly

import globals as g


# from utils.torch_utils import select_device
# from models.experimental import attempt_load
# from utils.general import check_img_size, non_max_suppression, scale_coords
# from utils.datasets import letterbox


CONFIDENCE = "confidence"
IMG_SIZE = 640


# def init_model(config, checkpoint=None, device='cuda:0', options=None):
#     """Initialize a classifier from config file.
#
#     Args:
#         config (str or :obj:`mmcv.Config`): Config file path or the config
#             object.
#         checkpoint (str, optional): Checkpoint path. If left as None, the model
#             will not load any weights.
#         options (dict): Options to override some settings in the used config.
#
#     Returns:
#         nn.Module: The constructed classifier.
#     """
#     if isinstance(config, str):
#         config = mmcv.Config.fromfile(config)
#     elif not isinstance(config, mmcv.Config):
#         raise TypeError('config must be a filename or Config object, '
#                         f'but got {type(config)}')
#     if options is not None:
#         config.merge_from_dict(options)
#     config.model.pretrained = None
#     model = build_classifier(config.model)
#     if checkpoint is not None:
#         map_loc = 'cpu' if device == 'cpu' else None
#         checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
#         if 'CLASSES' in checkpoint['meta']:
#             model.CLASSES = checkpoint['meta']['CLASSES']
#         else:
#             from mmcls.datasets import ImageNet
#             warnings.simplefilter('once')
#             warnings.warn('Class names are not saved in the checkpoint\'s '
#                           'meta data, use imagenet by default.')
#             model.CLASSES = ImageNet.CLASSES
#     model.cfg = config  # save the config in the model for convenience
#     model.to(device)
#     model.eval()
#     return model


@sly.timeit
def download_model_and_configs():
    if not g.remote_weights_path.endswith(".pth"):
        raise ValueError(f"Unsupported weights extension {sly.fs.get_file_ext(g.remote_weights_path)}. "
                         f"Supported extension: '.pth'")

    info = g.api.file.get_info_by_path(g.team_id, g.remote_weights_path)
    if info is None:
        raise FileNotFoundError(f"Weights file not found: {g.remote_weights_path}")

    progress = sly.Progress("Downloading weights", info.sizeb, is_size=True, need_info_log=True)
    g.local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(g.remote_weights_path))
    g.api.file.download(
        g.team_id,
        g.remote_weights_path,
        g.local_weights_path,
        cache=g.my_app.cache,
        progress_cb=progress.iters_done_report
    )

    def _download_dir(remote_dir, local_dir):
        remote_files = g.api.file.list2(g.team_id, remote_dir)
        progress = sly.Progress(f"Downloading {remote_dir}", len(remote_files), need_info_log=True)
        for remote_file in remote_files:
            local_file = os.path.join(local_dir, sly.fs.get_file_name_with_ext(remote_file.path))
            if sly.fs.file_exists(local_file):  # @TODO: for debug
                pass
            else:
                g.api.file.download(g.team_id, remote_file.path, local_file)
            progress.iter_done_report()

    _download_dir(g.remote_configs_dir, g.local_configs_dir)
    _download_dir(g.remote_info_dir, g.local_info_dir)

    sly.logger.info("Model has been successfully downloaded")


def construct_model_meta():
    g.labels_urls = sly.json.load_json_file(g.local_labels_urls_path)
    g.gt_labels = sly.json.load_json_file(g.local_gt_labels_path)
    tag_metas = []
    for name, index in g.gt_labels.items():
        tag_metas.append(sly.TagMeta(name, sly.TagValueType.NONE))
    g.meta = sly.ProjectMeta(tag_metas=sly.TagMetaCollection(tag_metas))




