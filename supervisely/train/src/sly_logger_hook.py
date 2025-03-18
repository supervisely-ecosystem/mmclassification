import datetime

import sly_globals as g
from mmengine.hooks import LoggerHook
from mmengine.registry import HOOKS
from sly_train_progress import (
    add_progress_to_request,
    get_progress_cb,
    set_progress,
)

import supervisely as sly

# from mmpretrain.registry import HOOKS


@HOOKS.register_module()
class SuperviselyLoggerHook(LoggerHook):
    def __init__(
        self, by_epoch=True, interval=10, ignore_last=True, reset_flag=False, interval_exp_name=1000
    ):
        super(SuperviselyLoggerHook, self).__init__(
            by_epoch, interval, ignore_last, reset_flag, interval_exp_name
        )
        self.progress_epoch = None
        self.progress_iter = None
        self._lrs = []

    def _log_info(self, log_dict, runner):
        super(SuperviselyLoggerHook, self)._log_info(log_dict, runner)

        log_dict["max_iters"] = runner.max_iters
        if log_dict["mode"] == "train" and "time" in log_dict.keys():
            temp = self.time_sec_tot + (log_dict["time"] * self.interval)
            time_sec_avg = temp / (runner.iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            log_dict["sly_eta"] = eta_str

        if self.progress_epoch is None:
            self.progress_epoch = sly.Progress("Epochs", runner.max_epochs)
        if self.progress_iter is None:
            self.progress_iter = sly.Progress("Iterations", len(runner.train_dataloader))

        fields = []
        if log_dict["mode"] == "val":
            self.progress_epoch.set_current_value(log_dict["epoch"])
            self.progress_iter.set_current_value(0)
        else:
            self.progress_iter.set_current_value(log_dict["iter"])
            fields.append({"field": "data.eta", "payload": log_dict["sly_eta"]})

        fields.append({"field": "state.isValidation", "payload": log_dict["mode"] == "val"})

        add_progress_to_request(fields, "Epoch", self.progress_epoch)
        add_progress_to_request(fields, "Iter", self.progress_iter)

        epoch_float = float(self.progress_epoch.current) + float(
            self.progress_iter.current
        ) / float(self.progress_iter.total)

        sly.logger.debug("Log dict", extra={"mm-train-logs": log_dict})
        if log_dict["mode"] == "train":
            fields.extend(
                [
                    {
                        "field": "data.chartLR.series[0].data",
                        "payload": [[epoch_float, round(log_dict["lr"], 6)]],
                        "append": True,
                    },
                    {
                        "field": "data.chartTrainLoss.series[0].data",
                        "payload": [[epoch_float, log_dict["loss"]]],
                        "append": True,
                    },
                ]
            )
            self._lrs.append(log_dict["lr"])
            fields.append(
                {
                    "field": "data.chartLR.options.yaxisInterval",
                    "payload": [
                        round(min(self._lrs) - min(self._lrs) / 10.0, 5),
                        round(max(self._lrs) + max(self._lrs) / 10.0, 5),
                    ],
                }
            )

            if "time" in log_dict.keys():
                fields.extend(
                    [
                        {
                            "field": "data.chartTime.series[0].data",
                            "payload": [[epoch_float, log_dict["time"]]],
                            "append": True,
                        },
                        {
                            "field": "data.chartDataTime.series[0].data",
                            "payload": [[epoch_float, log_dict["data_time"]]],
                            "append": True,
                        },
                        {
                            "field": "data.chartMemory.series[0].data",
                            "payload": [[epoch_float, log_dict["memory"]]],
                            "append": True,
                        },
                    ]
                )
        if log_dict["mode"] == "val":
            multi_label = True
            for key in log_dict.keys():
                if key.startswith("accuracy") or key.startswith("f1_score"):
                    multi_label = False
            if multi_label:
                fields.extend(
                    [
                        {
                            "field": "data.chartC.series[0].data",
                            "payload": [[log_dict["epoch"], log_dict["CP"]]],
                            "append": True,
                        },
                        {
                            "field": "data.chartC.series[1].data",
                            "payload": [[log_dict["epoch"], log_dict["CR"]]],
                            "append": True,
                        },
                        {
                            "field": "data.chartC.series[2].data",
                            "payload": [[log_dict["epoch"], log_dict["CF1"]]],
                            "append": True,
                        },
                        {
                            "field": "data.chartO.series[0].data",
                            "payload": [[log_dict["epoch"], log_dict["OP"]]],
                            "append": True,
                        },
                        {
                            "field": "data.chartO.series[1].data",
                            "payload": [[log_dict["epoch"], log_dict["OR"]]],
                            "append": True,
                        },
                        {
                            "field": "data.chartO.series[2].data",
                            "payload": [[log_dict["epoch"], log_dict["OF1"]]],
                            "append": True,
                        },
                        {
                            "field": "data.chartMAP.series[0].data",
                            "payload": [[log_dict["epoch"], log_dict["mAP"]]],
                            "append": True,
                        },
                    ]
                )
            else:
                fields.extend(
                    [
                        {
                            "field": "data.chartValF1.series[0].data",
                            "payload": [[log_dict["epoch"], log_dict["f1_score"]]],
                            "append": True,
                        },
                    ]
                )

        g.api.app.set_fields(g.task_id, fields)
