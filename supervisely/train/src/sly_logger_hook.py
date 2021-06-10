import datetime
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger.text import TextLoggerHook
from sly_train_progress import get_progress_cb, set_progress


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
        self.progress_iter_set = None


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
            self.progress_epoch = get_progress_cb("Epoch", "Epoch", runner.max_epochs, min_report_percent=1)
        if self.progress_iter_set is None:
            self.progress_iter_set = get_progress_cb("Iter", "Iterations", runner.max_iters, min_report_percent=1, upd_func=set_progress)
        self.progress_iter_set(log_dict['iter'])

        if log_dict['mode'] == 'val':
            #if runner.iter == runner.max_iters:
            self.progress_epoch(1)



        #download_progress = get_progress_cb(progress_index, "Download project", g.project_info.items_count * 2)

        # epoch progress
        # iter progress
        # : eta when training will be finished string
        # ETA gives an estimate of roughly how long the whole training process will take

        # train
        # charts: lr, time, data_time, memory, loss

        # val
        # charts: accuracy (accuracy_top-1, accuracy_top-5)