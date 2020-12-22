import copy
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

from dataloaders import CurrencyData, IEXDataset, collate_batch_fn
from models import EncoderDecoderLSTM


def now_str():
    return datetime.now().strftime('%b-%d-%H-%M-%S')


def today_str():
    return datetime.now().strftime('%Y-%b-%d')


logger = None
def make_logger(file_path:str):
    global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=file_path,
                        filemode='w',
                        level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


class SequencePredictorTrainer():
    def __init__(self, config, cl_args):
        super(SequencePredictorTrainer, self).__init__()
        for cfg, val in config.items():
            setattr(self, '_' + cfg, val)

        self._chckpt_dir = os.path.join(self._chckpt_dir, now_str())
        pathlib.Path(self._chckpt_dir).mkdir(parents=True, exist_ok=True)

        make_logger(os.path.join(self._chckpt_dir, 'trainer.log'))

        with open(os.path.join(self._chckpt_dir, 'config.json'), 'w') as out_config:
            json.dump({"config": config}, out_config, indent=2)

        train_dataset_args = config['dataset_config']
        model_args = config['model_config']
        assert ('country' in config) != ('train_tickers' in config)
        if 'country' in config:
            model_args['ip_n_feat'] = 1
            model_args['op_n_feat'] = 1
            dataset_class = CurrencyData
            train_dataset_args['country'] = self._country.upper()
            model_args['batch_n'] = self._batch_size
            self._name = self._country
        elif 'train_tickers' in config and 'valdn_tickers' in config:
            assert 'input_features' in train_dataset_args and 'output_features' in train_dataset_args
            model_args['ip_n_feat'] = len(train_dataset_args['input_features'])
            model_args['op_n_feat'] = len(train_dataset_args['output_features'])
            model_args['batch_n'] = self._batch_size
            train_dataset_args['tickers'] = self._train_tickers
            train_dataset_args['shuffle_rows'] = True
            valdn_dataset_args = copy.copy(train_dataset_args)
            valdn_dataset_args['tickers'] = self._valdn_tickers
            valdn_dataset_args['shuffle_rows'] = False
            dataset_class = IEXDataset
            self._name = self._valdn_tickers
        else:
            raise ValueError('model_config must have either "train_tickers" & "valdn_tickers"  or "country"')

        self._num_predictions = train_dataset_args['output_seq_len']

        self._model = EncoderDecoderLSTM(**model_args)

        self._optim = torch.optim.Adadelta(**config["optimizer_config"], params=self._model.parameters())

        self._lr_sched = torch.optim.lr_scheduler.StepLR(**config["lr_sched_config"], optimizer=self._optim)
        self._writer = SummaryWriter(self._chckpt_dir)

        # used in plotting
        self._output_features = train_dataset_args['output_features']

        train_dataset = dataset_class(cl_args, **train_dataset_args,
                                      num_days_to_fetch=self._num_train_days,
                                      num_days_to_skip=self._num_valdn_days)

        self._train_data = dataloader.DataLoader(train_dataset,
                                                 batch_size=self._batch_size,
                                                 drop_last=True,
                                                 shuffle=True,
                                                 collate_fn=collate_batch_fn)

        valdn_dataset = dataset_class(cl_args, **valdn_dataset_args,
                                      num_days_to_skip=0,
                                      num_days_to_fetch=self._num_valdn_days)

        self._valdn_data = dataloader.DataLoader(valdn_dataset,
                                                 batch_size=self._batch_size,
                                                 drop_last=True,
                                                 shuffle=True,
                                                 collate_fn=collate_batch_fn)
        logger.info(f'Training with  {len(self._train_data)} batches')
        logger.info(f'Validating with {len(self._valdn_data)} batches')

        loss_reduction = config['loss_reduction'] if 'loss_reduction' in config else 'mean'
        self._loss = nn.L1Loss(reduction=loss_reduction)

        self._it = self.epoch = 0

        self._valdn_data_scale_fn = valdn_dataset.scale

    def train(self):

        best_valdn_loss = float('inf')
        is_best = False

        def _train_one_sample(x, y):
            model_ip = (x, self._num_predictions)
            y_hat = self._model(model_ip)
            loss = self._loss(y_hat, y)

            self._optim.zero_grad()
            loss.backward()

            self._optim.step()
            self._lr_sched.step()

            loss_list.append(loss)
            self._it += 1

        @torch.no_grad()
        def _log_to_tb():
            mean_loss = torch.Tensor(loss_list).mean()
            self._writer.add_scalars('Loss', {'Train': mean_loss}, self._it)

            lr = self._optim.param_groups[0]['lr']
            self._writer.add_scalar('Optim/LR', lr, self._it)
            for n, p in self._model.named_parameters():
                self._writer.add_histogram(n + '_grads', p.grad, self._it)
                self._writer.add_histogram(n + '_value', p.data, self._it)

        self._it = 0
        for self.epoch in range(1, self._num_epochs + 1):
            loss_list = []
            self._model.train()
            for sample in iter(self._train_data):

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
        for (x, y) in iter(self._valdn_data):
            model_ip = (x, self._num_predictions)
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
        filename = os.path.join(self._chckpt_dir,  suffix + '.pt')
        torch.save({'model' : self._model,
                    'epoch': self.epoch,
                    'optim': self._optim.state_dict(),
                    'lr_sched': self._lr_sched.state_dict()}, filename)

    @torch.no_grad()
    def resume_checkpoint(self, filename):
        if not os.path.exists(filename):
            raise RuntimeError('Invalid checkpoint location: {}'.format(filename))

        if not os.path.exists(filename):
            raise RuntimeError('Did not find any valid checkpoints in {}.'.format(filename))
        logger.debug('Loading checkpoint at {}'.format(filename))
        checkpoint = torch.load(filename)

        self._model.load_state_dict(checkpoint['model'])
        self._optim.load_state_dict(checkpoint['optim'])
        self._lr_sched.load_state_dict(checkpoint['lr_sched'])

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
                save_path = os.path.join(self._chckpt_dir, title + '-EP_' + str(self.epoch) + '.png')
                plt.savefig(save_path, dpi=100)
                save_paths.append(save_path)
                logger.debug(f'Plot image saved to {save_path}')
            plt.clf()
        return save_paths
