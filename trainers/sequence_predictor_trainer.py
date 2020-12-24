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

logger = logging.getLogger(__name__)


def collate_batch_fn(batch):
    xs = [i[0] for i in batch]
    ys = [i[1] for i in batch]
    return tuple([torch.stack(xs), torch.stack(ys)])


class SequencePredictorTrainer():
    """Train 1D sequences."""

    def __init__(self, train_dataset, valdn_dataset, **kwargs):
        super().__init__()
        for cfg, val in kwargs.items():
            setattr(self, '_' + cfg, val)

        self._name = train_dataset.name

        self._train_dataloader = dataloader.DataLoader(train_dataset,
                                                       batch_size=self._batch_size,
                                                       drop_last=True,
                                                       shuffle=True,
                                                       collate_fn=collate_batch_fn)

        self._valdn_dataloader = dataloader.DataLoader(valdn_dataset,
                                                       batch_size=self._batch_size,
                                                       drop_last=True,
                                                       collate_fn=collate_batch_fn)

        logger.info(f'Training on {len(self._train_dataloader)} batches, and validating on \
                     {len(self._valdn_dataloader)} batches')

        self._writer = SummaryWriter(self._logging_dir)
        self._valdn_data_scale_fn = valdn_dataset.scale
        self._it = self.epoch = 0

    def train(self):

        best_valdn_loss = float('inf')
        is_best = False

        def _train_one_sample(x, y):
            model_ip = (x, self._output_seq_len)
            y_hat = self._model(model_ip)
            loss = self._loss(y_hat, y)

            self._optimizer.zero_grad()
            loss.backward()

            self._optimizer.step()
            self._lr_scheduler.step()

            loss_list.append(loss.item())
            self._it += 1

        @torch.no_grad()
        def _log_to_tb():
            mean_loss = torch.Tensor(loss_list).mean()
            self._writer.add_scalars('Loss', {'Train': mean_loss}, self._it)

            lr = self._optimizer.param_groups[0]['lr']
            self._writer.add_scalar('Optim/LR', lr, self._it)
            for n, p in self._model.named_parameters():
                self._writer.add_histogram(n + '_grads', p.grad, self._it)
                self._writer.add_histogram(n + '_value', p.data, self._it)

        self._it = 0
        for self.epoch in range(1, self._num_epochs + 1):
            loss_list = []
            self._model.train()
            for sample in iter(self._train_dataloader):

                _train_one_sample(*sample)

                if self._it % self._log_n_iter == 0:
                    _log_to_tb()

            mean_loss = torch.Tensor(loss_list).mean()
            logger.info(f'Epoch: {self.epoch:03} Training Avg-loss: {mean_loss:2.3}')

            if self.epoch % self._valdn_n_epochs == 0 and self.epoch > 0:
                valdn_loss, _ = self.run_validation(show_plot=False, save_plot=True)
                is_best = valdn_loss.item() < best_valdn_loss
                if is_best:
                    best_valdn_loss = valdn_loss
                logger.info(f'Epoch: {self.epoch:03} --> Validation Avg-loss: {valdn_loss:2.3}' +
                            (' <--' if is_best else ', Best: ' + str(best_valdn_loss.item())))

            if self.epoch % (self._checkpt_n_valdn * self._valdn_n_epochs) == 0:
                self.save_checkpoint(is_best=is_best)

        self.run_validation(True, True)
        self.save_checkpoint(is_best=is_best)

    @torch.no_grad()
    def run_validation(self, show_plot=False, save_plot=False):
        loss_list, pred_list, tgt_list = [], [], []
        self._model.eval()
        for (x, y) in iter(self._valdn_dataloader):
            model_ip = (x, self._output_seq_len)
            y_hat = self._model(model_ip)
            loss = self._loss(y_hat, y)
            loss_list.append(loss)
            pred_list.append(y_hat)
            tgt_list.append(y)
        mean_loss = torch.Tensor(loss_list).mean()
        self._writer.add_scalars('Loss', {'Validation': mean_loss}, self._it)

        save_paths = self._plot_valdn(pred_list, tgt_list, show=show_plot, save=save_plot)
        return mean_loss, save_paths

    @torch.no_grad()
    def save_checkpoint(self, is_best=False):
        suffix = self._name + '-EP_' + str(self.epoch)
        if is_best:
            suffix += '_better'
        filename = os.path.join(self._logging_dir,  suffix + '.pt')
        torch.save({'model': self._model,
                    'epoch': self.epoch,
                    'optim': self._optimizer.state_dict(),
                    'lr_sched': self._lr_scheduler.state_dict()}, filename)

    @torch.no_grad()
    def resume_checkpoint(self, filename):
        if not os.path.exists(filename):
            raise RuntimeError('Invalid checkpoint location: {}'.format(filename))

        if not os.path.exists(filename):
            raise RuntimeError('Did not find any valid checkpoints in {}.'.format(filename))
        logger.debug('Loading checkpoint at {}'.format(filename))
        checkpoint = torch.load(filename)

        self._model.load_state_dict(checkpoint['model'])
        self._optimizer.load_state_dict(checkpoint['optim'])
        self._lr_scheduler.load_state_dict(checkpoint['lr_sched'])

    @torch.no_grad()
    def _plot_valdn(self, pred_slid_win, tgt_slid_win, show=False, save=False):

        save_paths = []

        if not show and not save:
            return save_paths

        def _sliding_win_to_seq(windows: list, component: int):
            windows = [torch.squeeze(w[:, :, component]) for w in windows]
            windows = torch.cat(windows, dim=0)
            ret = windows
            if windows.dim() == 2:
                ret = list(windows[0]) + list(windows[1:, 1])
            return torch.tensor([r.item() for r in ret])

        for idx, op_feat in enumerate(self._output_features):
            tgt_list = _sliding_win_to_seq(tgt_slid_win, idx)
            pred_list = _sliding_win_to_seq(pred_slid_win, idx)
            _, pred_list = self._valdn_data_scale_fn(None, pred_list)
            _, tgt_list = self._valdn_data_scale_fn(None, tgt_list)
            plt.plot(pred_list, label='Predicted')
            plt.plot(tgt_list, label='Actual')
            title = self._name.upper() + '-' + op_feat.capitalize()
            plt.title(title)
            plt.legend()
            if show:
                plt.show()
                logger.info('Actual:   ' + str([f'{x:.3f} ' for x in tgt_list]))
                logger.info('Predicted:' + str([f'{x:1.3f} ' for x in pred_list]))
            if save:
                save_path = os.path.join(self._logging_dir, title + '-EP_' + str(self.epoch) + '.png')
                plt.savefig(save_path, dpi=100)
                save_paths.append(save_path)
                logger.debug(f'Plot image saved to {save_path}')
            plt.clf()
        return save_paths
