import datetime
from collections import OrderedDict

import torch
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger.text import TextLoggerHook


@HOOKS.register_module()
class SuperviselyLoggerHook(TextLoggerHook):
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

        # epoch progress
        # iter progress
        # : eta when training will be finished string
        # ETA gives an estimate of roughly how long the whole training process will take

        # train
        # charts: lr, time, data_time, memory, loss

        # val
        # charts: accuracy (accuracy_top-1, accuracy_top-5)