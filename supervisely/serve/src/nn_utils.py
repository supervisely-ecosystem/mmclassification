import os

import globals as g
import mmcv
import numpy as np
import torch
from mmpretrain.apis import init_model
from mmpretrain.datasets.transforms import Compose
from mmcv.parallel import collate, scatter

import supervisely as sly


@sly.timeit
def download_model_and_configs():
    if not g.remote_weights_path.endswith(".pth"):
        raise ValueError(
            f"Unsupported weights extension {sly.fs.get_file_ext(g.remote_weights_path)}. "
            f"Supported extension: '.pth'"
        )

    info = g.api.file.get_info_by_path(g.team_id, g.remote_weights_path)
    if info is None:
        raise FileNotFoundError(f"Weights file not found: {g.remote_weights_path}")

    progress = sly.Progress("Downloading weights", info.sizeb, is_size=True, need_info_log=True)
    g.local_weights_path = os.path.join(
        g.my_app.data_dir, sly.fs.get_file_name_with_ext(g.remote_weights_path)
    )
    g.api.file.download(
        g.team_id,
        g.remote_weights_path,
        g.local_weights_path,
        cache=g.my_app.cache,
        progress_cb=progress.iters_done_report,
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
    g.gt_index_to_labels = {index: name for name, index in g.gt_labels.items()}

    tag_metas = []
    for name, index in g.gt_labels.items():
        tag_metas.append(sly.TagMeta(name, sly.TagValueType.NONE))
    g.meta = sly.ProjectMeta(tag_metas=sly.TagMetaCollection(tag_metas))


@sly.timeit
def deploy_model():
    cfg = mmcv.Config.fromfile(g.local_model_config_path)
    if hasattr(cfg, "classification_mode"):
        g.cls_mode = cfg.classification_mode
    # g.model = init_model(cfg, g.local_weights_path, device=g.device)
    g.model = init_model(cfg, g.local_weights_path, device="cpu")

    g.model.CLASSES = sorted(g.gt_labels, key=g.gt_labels.get)
    sly.logger.info("🟩 Model has been successfully deployed")


def inference_model(model, img, topn=5):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (list of dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]["type"] != "LoadImageFromFile":
            cfg.data.test.pipeline.insert(0, dict(type="LoadImageFromFile"))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]["type"] == "LoadImageFromFile":
            cfg.data.test.pipeline.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        scores = model(return_loss=False, **data)
        model_out = scores[0]
        result = []
        if topn is None:  # multi-label
            top_labels = model_out.argsort()  # [::-1]
            top_labels = top_labels[model_out[top_labels] > 0.5][::-1]
            top_scores = model_out[top_labels]
        else:  # one-label with top-n
            top_scores = model_out[model_out.argsort()[-topn:]][::-1]
            top_labels = model_out.argsort()[-topn:][::-1]

        for label, score in zip(top_labels, top_scores):
            result.append(
                {"label": int(label), "score": float(score), "class": model.CLASSES[label]}
            )
    return result


def inference_model_batch(model, images_nps, topn=5):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (list of dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if cfg.data.test.pipeline[0]["type"] == "LoadImageFromFile":
        cfg.data.test.pipeline.pop(0)

    test_pipeline = Compose(cfg.data.test.pipeline)

    with torch.no_grad():

        inference_results = []
        for images_batch in sly.batched(images_nps, g.batch_size):
            data = [dict(img=img) for img in images_batch]

            data = [test_pipeline(row) for row in data]
            data = collate(data, samples_per_gpu=1)

            if next(model.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [device])[0]

            batch_scores = np.asarray(model(return_loss=False, **data))
            if topn is not None:  # one-label with top-n
                batch_top_indexes = batch_scores.argsort(axis=1)[:, -topn:][:, ::-1]
                for scores, top_indexes in zip(batch_scores, batch_top_indexes):
                    inference_results.append(
                        {
                            "label": top_indexes.astype(int).tolist(),
                            "score": scores[top_indexes].astype(float).tolist(),
                            "class": np.asarray(model.CLASSES)[top_indexes].tolist(),
                        }
                    )
            else:  # multi-label
                batch_top_indexes = batch_scores.argsort(axis=1)

                batch_top_indexes = [
                    sample_inds[batch_scores[i][sample_inds] > 0.5][::-1]
                    for i, sample_inds in enumerate(batch_top_indexes)
                ]
                for scores, top_indexes in zip(batch_scores, batch_top_indexes):
                    inference_results.append(
                        {
                            "label": top_indexes.astype(int).tolist(),
                            "score": scores[top_indexes].astype(float).tolist(),
                            "class": np.asarray(model.CLASSES)[top_indexes].tolist(),
                        }
                    )

    return inference_results
