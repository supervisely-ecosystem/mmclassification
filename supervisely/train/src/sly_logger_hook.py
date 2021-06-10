import datetime
from collections import OrderedDict

import torch
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger.text import TextLoggerHook



@HOOKS.register_module()
class SuperviselyLoggerHook(TextLoggerHook):

    def log(self, runner):
        if 'eval_iter_num' in runner.log_buffer.output:
            # this doesn't modify runner.iter and is regardless of by_epoch
            cur_iter = runner.log_buffer.output.pop('eval_iter_num')
        else:
            cur_iter = self.get_iter(runner, inner_iter=True)

        log_dict = OrderedDict(
            mode=self.get_mode(runner),
            epoch=self.get_epoch(runner),
            iter=cur_iter)

        # only record lr of the first param group
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            log_dict['lr'] = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            log_dict['lr'] = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                log_dict['lr'].update({k: lr_[0]})

        if 'time' in runner.log_buffer.output:
            # statistic memory
            if torch.cuda.is_available():
                log_dict['memory'] = self._get_max_memory(runner)

        log_dict = dict(log_dict, **runner.log_buffer.output)
        log_dict['max_iters'] = runner.max_iters

        if log_dict['mode'] == 'train' and 'time' in log_dict.keys():
            temp = self.time_sec_tot + (log_dict['time'] * self.interval)
            time_sec_avg = temp / (runner.iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            log_dict['sly_eta'] = eta_str
            log_dict['sly_time'] = f'{log_dict["time"]:.3f}'
            log_dict['sly_data_time'] = f'{log_dict["data_time"]:.3f}'

        self._log_info(log_dict, runner)
        self._dump_log(log_dict, runner)
        self._send_log(log_dict)

    def send_log(self, log_dict):
        pass