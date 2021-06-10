import datetime
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger.text import TextLoggerHook
import supervisely_lib as sly
from sly_train_progress import get_progress_cb, set_progress, add_progress_to_request
import sly_globals as g


@HOOKS.register_module()
class SuperviselyLoggerHook(TextLoggerHook):
    def __init__(self,
                 by_epoch=True,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 interval_exp_name=1000):
        super(SuperviselyLoggerHook, self).__init__(by_epoch, interval, ignore_last, reset_flag, interval_exp_name)
        self.progress_epoch = None
        self.progress_iter = None


    def _log_info(self, log_dict, runner):
        super(SuperviselyLoggerHook, self)._log_info(log_dict, runner)

        log_dict['max_iters'] = runner.max_iters
        if log_dict['mode'] == 'train' and 'time' in log_dict.keys():
            temp = self.time_sec_tot + (log_dict['time'] * self.interval)
            time_sec_avg = temp / (runner.iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            log_dict['sly_eta'] = eta_str
            #log_dict['sly_time'] = f'{log_dict["time"]:.3f}'
            #log_dict['sly_data_time'] = f'{log_dict["data_time"]:.3f}'
        pass

        if self.progress_epoch is None:
            self.progress_epoch = sly.Progress("Epochs", runner.max_epochs)
        if self.progress_iter is None:
            self.progress_iter = sly.Progress("Iterations", len(runner.data_loader))

        fields = []
        if log_dict['mode'] == 'val':
            self.progress_epoch.set_current_value(log_dict["epoch"])
            self.progress_iter.set_current_value(0)
        else:
            self.progress_iter.set_current_value(log_dict['iter'])
            fields.append({"field": "data.eta", "payload": log_dict['sly_eta']})

        add_progress_to_request(fields, "Epoch", self.progress_epoch)
        add_progress_to_request(fields, "Iter", self.progress_iter)
        g.api.app.set_fields(g.task_id, fields)

        #download_progress = get_progress_cb(progress_index, "Download project", g.project_info.items_count * 2)

        # epoch progress
        # iter progress
        # : eta when training will be finished string
        # ETA gives an estimate of roughly how long the whole training process will take

        # train
        # charts: lr, time, data_time, memory, loss

        # val
        # charts: accuracy (accuracy_top-1, accuracy_top-5)