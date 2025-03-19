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
        sly.logger.info("SuperviselyLoggerHook initialized", extra={"task_id": g.task_id})

    def _log_info(self, log_dict, runner):
        super(SuperviselyLoggerHook, self)._log_info(log_dict, runner)

        # Detailed log of incoming data
        sly.logger.debug("Log dict received", extra={"log_dict": log_dict})
        
        log_dict["max_iters"] = runner.max_iters
        if log_dict["mode"] == "train" and "time" in log_dict.keys():
            temp = self.time_sec_tot + (log_dict["time"] * self.interval)
            time_sec_avg = temp / (runner.iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            log_dict["sly_eta"] = eta_str

        # Debug progress information
        sly.logger.debug(
            "Progress info", 
            extra={
                "max_epochs": runner.max_epochs,
                "train_dataloader_len": len(runner.train_dataloader),
            }
        )
        
        if self.progress_epoch is None:
            self.progress_epoch = sly.Progress("Epochs", runner.max_epochs)
            sly.logger.debug("Created progress_epoch", extra={"total": runner.max_epochs})
        if self.progress_iter is None:
            self.progress_iter = sly.Progress("Iterations", len(runner.train_dataloader))
            sly.logger.debug("Created progress_iter", extra={"total": len(runner.train_dataloader)})

        fields = []
        if log_dict["mode"] == "val":
            self.progress_epoch.set_current_value(log_dict["epoch"])
            self.progress_iter.set_current_value(0)
            sly.logger.debug("Validation mode progress update", extra={"current_epoch": log_dict["epoch"]})
        else:
            self.progress_iter.set_current_value(log_dict["iter"])
            fields.append({"field": "data.eta", "payload": log_dict["sly_eta"]})
            sly.logger.debug("Train mode progress update", extra={"current_iter": log_dict["iter"]})

        fields.append({"field": "state.isValidation", "payload": log_dict["mode"] == "val"})

        # Check before adding progress
        sly.logger.debug("Progress before add_progress_to_request", 
                        extra={
                            "epoch_progress": {
                                "current": self.progress_epoch.current,
                                "total": self.progress_epoch.total
                            },
                            "iter_progress": {
                                "current": self.progress_iter.current,
                                "total": self.progress_iter.total
                            }
                        })
                        
        add_progress_to_request(fields, "Epoch", self.progress_epoch)
        add_progress_to_request(fields, "Iter", self.progress_iter)
        
        # Check after adding progress
        sly.logger.debug("Progress fields after add_progress_to_request", extra={"fields": fields})

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
            
            sly.logger.debug("Validation metrics detection", extra={
                "multi_label": multi_label,
                "keys": list(log_dict.keys())
            })
                
            if multi_label:
                # Check for required keys
                required_keys = ["CP", "CR", "CF1", "OP", "OR", "OF1", "mAP"]
                missing_keys = [key for key in required_keys if key not in log_dict]
                if missing_keys:
                    sly.logger.warning(f"Missing validation metrics keys: {missing_keys}")
                    
                fields.extend(
                    [
                        {
                            "field": "data.chartC.series[0].data",
                            "payload": [[log_dict["epoch"], log_dict.get("CP", 0)]],
                            "append": True,
                        },
                        {
                            "field": "data.chartC.series[1].data",
                            "payload": [[log_dict["epoch"], log_dict.get("CR", 0)]],
                            "append": True,
                        },
                        {
                            "field": "data.chartC.series[2].data",
                            "payload": [[log_dict["epoch"], log_dict.get("CF1", 0)]],
                            "append": True,
                        },
                        {
                            "field": "data.chartO.series[0].data",
                            "payload": [[log_dict["epoch"], log_dict.get("OP", 0)]],
                            "append": True,
                        },
                        {
                            "field": "data.chartO.series[1].data",
                            "payload": [[log_dict["epoch"], log_dict.get("OR", 0)]],
                            "append": True,
                        },
                        {
                            "field": "data.chartO.series[2].data",
                            "payload": [[log_dict["epoch"], log_dict.get("OF1", 0)]],
                            "append": True,
                        },
                        {
                            "field": "data.chartMAP.series[0].data",
                            "payload": [[log_dict["epoch"], log_dict.get("mAP", 0)]],
                            "append": True,
                        },
                    ]
                )
            else:
                # Check for f1_score key
                if "f1_score" not in log_dict:
                    sly.logger.warning("Missing f1_score key in validation metrics")
                    
                fields.extend(
                    [
                        {
                            "field": "data.chartValF1.series[0].data",
                            "payload": [[log_dict["epoch"], log_dict.get("f1_score", 0)]],
                            "append": True,
                        },
                    ]
                )

        # Check before API call
        sly.logger.debug("Before API call", extra={
            "task_id": g.task_id,
            "fields_count": len(fields),
        })
        
        try:
            g.api.app.set_fields(g.task_id, fields)
            sly.logger.debug("API call successful")
        except Exception as e:
            sly.logger.error(f"Error setting fields via API: {str(e)}", exc_info=True)
            
    def before_train(self, runner):
        super().before_train(runner)
        sly.logger.info("Training started", extra={
            "max_epochs": runner.max_epochs,
            "max_iters": runner.max_iters,
        })
