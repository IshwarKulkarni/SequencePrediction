import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data.dataloader as dataloader
from dataset_loaders.base_dataset import collate_batch_fn, TrainableDataset
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class SequencePredictorTrainer():
    """Train 1D sequences."""

    def __init__(self, train_dataset: TrainableDataset, valdn_dataset: TrainableDataset, **kwargs):
        super().__init__()
        for cfg, val in kwargs.items():
            setattr(self, '_' + cfg, val)

        self._it = self.epoch = 0
        if 'resume' in kwargs:
            self.resume_checkpoint(kwargs['resume'])

        self._name = valdn_dataset.name

        self._train_dataloader = dataloader.DataLoader(train_dataset,
                                                       batch_size=self._batch_size,
                                                       shuffle=True,
                                                       collate_fn=collate_batch_fn)

        self._valdn_dataloader = dataloader.DataLoader(valdn_dataset, batch_size=3)

        logger.info(f'Training on {len(self._train_dataloader)} batches, and validating on '
                    f'{len(self._valdn_dataloader)} sequences')

        self._writer = SummaryWriter(self._logging_dir) if self._log_tensorboard else None

        self._valdn_scale_fn = valdn_dataset.scale

    def train(self):

        best_valdn_loss, best_epoch = float('inf'), -1
        is_better = False

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
            for x, y in iter(self._train_dataloader):

                model_ip = (x, self._output_seq_len)
                y_hat = self._model(model_ip)
                loss = self._loss(y_hat[-self._output_seq_len:], y[-self._output_seq_len:])

                self._optimizer.zero_grad()
                loss.backward()

                self._optimizer.step()
                self._lr_scheduler.step()

                loss_list.append(loss.item())
                self._it += 1

                if self._log_tensorboard and self._it % self._log_n_iter == 0:
                    _log_to_tb()

            mean_loss = torch.Tensor(loss_list).mean()
            logger.info(f'Epoch: {self.epoch:03} Training Avg-loss: {mean_loss:2.3}')

            if self.epoch % self._valdn_n_epochs == 0 and self.epoch > 0:

                valdn_loss, _ = self.run_validation(show_plot=False, save_plot=True)
                is_better = valdn_loss.item() < best_valdn_loss

                if is_better:
                    best_valdn_loss, best_epoch = valdn_loss, self.epoch
                best_str = '<--' if is_better else f', Best: {best_valdn_loss.item():2.4} @ Epoch {best_epoch}'
                logger.info(f'Epoch: {self.epoch:03} --> Validation Avg-loss: {valdn_loss:2.3} {best_str}')

            if self.epoch % (self._checkpt_n_valdn * self._valdn_n_epochs) == 0:
                self.save_checkpoint(is_better=is_better)

        self.run_validation(True, True)
        self.save_checkpoint(is_better=is_better)

    @torch.no_grad()
    def run_validation(self, show_plot, save_plot):
        loss_list, pred_list, tgt_list = [], [], []
        self._model.eval()
        for (x, y) in iter(self._valdn_dataloader):
            model_ip = (x, self._output_seq_len)
            y_hat = self._model(model_ip)
            loss = self._loss(y_hat[-self._output_seq_len:], y[-self._output_seq_len:])
            loss_list.append(loss)
            pred_list.append(y_hat)
            tgt_list.append(y)
        mean_loss = torch.Tensor(loss_list).mean()
        if self._log_tensorboard:
            self._writer.add_scalars('Loss', {'Validation': mean_loss}, self._it)

        self._plot_valdn(pred_list, tgt_list, show=show_plot, save=save_plot)
        self._model.train()
        return mean_loss, []

    @torch.no_grad()
    def save_checkpoint(self, is_better=False):
        def save(filename):
            torch.save({'model': self._model.state_dict(),
                        'optim': self._optimizer.state_dict(),
                        'lr_sched': self._lr_scheduler.state_dict()}, filename)
        save(os.path.join(self._logging_dir, self._name + '-EP_' + str(self.epoch) + '.pt'))
        if is_better:
            save(os.path.join(self._logging_dir, self._name + '_best.pt'))

    @torch.no_grad()
    def resume_checkpoint(self, filename):
        assert os.path.exists(filename), f'Did not find any valid checkpoints in {filename}.'
        logger.debug(f'Loading checkpoint at {filename}')

        checkpoint = torch.load(filename)

        self._model.load_state_dict(checkpoint['model'])
        self._optimizer.load_state_dict(checkpoint['optim'])
        self._lr_scheduler.load_state_dict(checkpoint['lr_sched'])

    @torch.no_grad()
    def _plot_valdn(self, pred, tgt, show, save):
        purple = (.60, .45, .90)
        orange = (.90, .40, .15)
        green = (.28, .90, .15)
        if not show and not save:
            return
        in_n, out_n = self._input_seq_len, self._output_seq_len
        _, tgt = self._valdn_scale_fn(None, torch.cat(tgt))
        _, pred = self._valdn_scale_fn(None, torch.cat(pred))

        tgt = tgt.reshape(-1, tgt.shape[-1])
        x = 0
        for i, op_feat in enumerate(self._output_features):
            plt.plot(tgt[..., i], label='Actual', linewidth=0.5, color=purple, linestyle='dotted')
            title = self._name.upper() + '_' + op_feat.capitalize()
            for seg in pred:
                plt.plot(range(x, x + in_n), seg[0:in_n], color=orange, linewidth=.5)
                x += in_n
                plt.plot(range(x-1, x + out_n), seg[(in_n-1):], color=green, linewidth=.5)
                x += out_n
            plt.plot([], [], color=orange, linewidth=.5, label='Predicted@Input')
            plt.plot([], [], color=green, linewidth=.5, label='Predicted@Output')
            plt.title(title)
            plt.legend()
            if save:
                save_path = os.path.join(self._logging_dir, title + '-EP_' + str(self.epoch) + '.png')
                plt.savefig(save_path, dpi=200)
                logger.debug(f'Plot image saved to {save_path}')
            if show:
                plt.show()
            plt.clf()
