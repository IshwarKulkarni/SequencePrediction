import json
import logging
import os
import pathlib
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
from torch.utils.tensorboard import SummaryWriter

from dataloaders import IEXDataset
from models import EncoderDecoderLSTM


def now_str():
    return datetime.now().strftime('%b-%d-%H-%M-%S')


def today_str():
    return datetime.now().strftime('%Y-%b-%d')


logger = logging.getLogger(__name__)
logging.basicConfig(filename='trainer.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


class SequencePredictorTrainer():
    def __init__(self, config_filename):

        with open(config_filename) as config_file:
            config = json.load(config_file)['trainer_config']

        for cfg, val in config.items():
            setattr(self, '_' + cfg, val)

        self._chckpt_dir = os.path.join(self._chckpt_dir, now_str())
        pathlib.Path(self._chckpt_dir).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(self._chckpt_dir, 'config.json'), 'w') as out_config:
            json.dump({"config": config}, out_config, indent=2)

        self._model = EncoderDecoderLSTM(batch_size=self._batch_size, input_features_sz=len(self._input_features),
                                         output_features_sz=len(self._output_features))

        self._optim = torch.optim.Adam(**config["optimizer_config"],
                                       params=self._model.parameters())

        self._lr_sched = torch.optim.lr_scheduler.StepLR(**config["lr_sched_config"],
                                                         optimizer=self._optim)
        self._loss = nn.MSELoss(reduction='mean')

        self._writer = SummaryWriter(self._chckpt_dir)

        train_dataset = IEXDataset(self._ticker,
                                   num_days_to_fetch=self._num_train_days,
                                   num_days_to_skip=self._num_valdn_days,
                                   input_features=self._input_features,
                                   output_features=self._output_features,
                                   input_seq_sz=self._in_seq_len,
                                   output_seq_size=self._out_seq_len)

        self._train_data = dataloader.DataLoader(train_dataset,
                                                 batch_size=self._batch_size,
                                                 drop_last=True,
                                                 num_workers=self._num_workers,
                                                 collate_fn=train_dataset.collate_batch)

        valdn_dataset = IEXDataset(self._ticker,
                                   num_days_to_fetch=self._num_valdn_days,
                                   num_days_to_skip=0,
                                   input_features=self._input_features,
                                   output_features=self._output_features,
                                   input_seq_sz=self._in_seq_len,
                                   output_seq_size=self._out_seq_len)

        self._valdn_data = dataloader.DataLoader(valdn_dataset,
                                                 batch_size=self._batch_size,
                                                 drop_last=True,
                                                 num_workers=self._num_workers,
                                                 collate_fn=valdn_dataset.collate_batch)

    def train(self):
        it = 0
        for e in range(0, self._num_epochs + 1):
            loss_list = []
            self._model.train()
            for (x, y) in iter(self._train_data):
                y_hat = self._model(x)
                loss = self._loss(
                    y_hat[:, -self._out_seq_len:], y[:, -self._out_seq_len:])

                self._optim.zero_grad()
                loss.backward()

                self._optim.step()
                self._lr_sched.step()

                loss_list.append(loss)
                it += 1

                if it % self._log_n_iter == 0:
                    mean_loss = torch.Tensor(loss_list).mean()
                    self._writer.add_scalars('Loss', {'Train': mean_loss}, it)
                    self._writer.add_scalar('Optim/LR', self._optim.param_groups[0]['lr'], it)
                    for n, p in self._model.named_parameters():
                        self._writer.add_histogram(n +'_grads', p.grad, it)
                        self._writer.add_histogram(n +'_value', p.data, it)

            mean_loss = torch.Tensor(loss_list).mean()
            logger.info(f'Epoch: {e:03} Training Avg-loss: {mean_loss:9.6}')

            if e % self._valdn_n_epochs == 0 and e > 0:
                self._run_validation(e, it)
            if e % self._checkpt_n_epochs == 0:
                self._save_checkpoint(e)

    def _run_validation(self, epoch, it):
        loss_list, pred_list, tgt_list = [], [], []
        self._model.eval()
        for (x, y) in iter(self._valdn_data):
            y_hat = self._model(x)[:, -self._out_seq_len:]
            loss = self._loss(y_hat, y[:, -self._out_seq_len:])
            loss_list.append(loss)
            pred_list.append(y_hat[:, -self._out_seq_len:])
            tgt_list.append(y[:, -self._out_seq_len:])
        mean_loss = torch.Tensor(loss_list).mean()
        self._writer.add_scalars('Loss', {'Validation': mean_loss}, it)
        logger.info(
            f'Epoch: {epoch:03} \t Validation Avg-loss: {mean_loss:9.6}')
        if epoch > 75 and False:
            self._show_plot(pred_list, tgt_list, epoch)

    def _save_checkpoint(self, epoch):
        pass

    def _show_plot(self, pred_slid_win, tgt_slid_win, epoch):

        def sliding_win_to_seq(windows: list, component: int):
            windows = [torch.squeeze(w[:, :, component]) for w in windows]
            windows = torch.cat(windows, dim=0)
            ret = list(windows[0] * 100) + list(windows[1:, 1] * 100)
            ret = [r.item() for r in ret]
            return ret

        for idx, op_feat in enumerate(self._output_features):
            tgt_list = sliding_win_to_seq(tgt_slid_win, idx)
            pred_list = sliding_win_to_seq(pred_slid_win, idx)
            x_axis = range(len(pred_list))
            plt.plot(pred_list, label='Predicted')
            plt.plot(tgt_list, label='Actual')
            plt.title(self._ticker + '-' + op_feat)
            plt.legend()
            plt.show()


if __name__ == "__main__":
    trainer = SequencePredictorTrainer('D:\\tickertot\\config.json')
    trainer.train()
