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

from dataloaders import IEXDataset, CurrencyData, collate_batch_fn
from models import EncoderDecoderLSTM


def now_str():
    return datetime.now().strftime('%b-%d-%H-%M-%S')


def today_str():
    return datetime.now().strftime('%Y-%b-%d')


logger = logging.getLogger(__name__)
logging.basicConfig(filename='trainer.log',
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

        with open(os.path.join(self._chckpt_dir, 'config.json'), 'w') as out_config:
            json.dump({"config": config}, out_config, indent=2)

        dataset_args = config['dataset_config']
        model_args = config['model_config']
        assert ('country' in config) != ('ticker' in config)
        if 'country' in config:
            model_args['ip_n_feat'] = 1
            model_args['op_n_feat'] = 1
            dataset_class = CurrencyData
            dataset_args['country'] = self._country.upper()
            self._name = self._country
        elif 'ticker' in config:
            assert 'input_features' in dataset_args and 'output_features' in dataset_args
            model_args['ip_n_feat'] = len(dataset_args['input_features'])
            model_args['op_n_feat'] = len(dataset_args['output_features'])
            dataset_args['ticker'] = self._ticker
            dataset_class = IEXDataset
            self._name = self._ticker
        else:
            raise ValueError('model_config must have either "ticker" or "country"')
    
        self._num_predictions = dataset_args['output_seq_len']

        self._model = EncoderDecoderLSTM(**model_args)

        self._optim = torch.optim.Adam(**config["optimizer_config"], params=self._model.parameters())

        self._lr_sched = torch.optim.lr_scheduler.StepLR(**config["lr_sched_config"], optimizer=self._optim)
        self._writer = SummaryWriter(self._chckpt_dir)

        # used in plotting
        self._output_features = dataset_args['output_features']

        train_dataset = dataset_class(cl_args, **dataset_args,
                                      num_days_to_fetch=self._num_train_days,
                                      num_days_to_skip=self._num_valdn_days)

        self._train_data = dataloader.DataLoader(train_dataset,
                                                 batch_size=self._batch_size,
                                                 drop_last=True,
                                                 collate_fn=collate_batch_fn)

        valdn_dataset = dataset_class(cl_args, **dataset_args,
                                      num_days_to_skip=0,
                                      num_days_to_fetch=self._num_valdn_days)

        self._valdn_data = dataloader.DataLoader(valdn_dataset,
                                                 batch_size=self._batch_size,
                                                 drop_last=True,
                                                 collate_fn=collate_batch_fn)
        logger.info(f'Training with  {len(self._train_data)} batches')
        logger.info(f'Validating with {len(self._valdn_data)} batches')

        loss_reduction = config['loss_reduction'] if 'loss_reduction' in config else 'mean'
        self._loss = nn.L1Loss(reduction=loss_reduction)

        self._it = self._epoch = 0

    def train(self):

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
            logger.info(f'Epoch: {self.epoch:03} Training Avg-loss: {mean_loss:9.6}')

            if self.epoch % self._valdn_n_epochs == 0 and self.epoch > 0:
                self.run_validation()
            if self.epoch % self._checkpt_n_epochs == 0:
                self.save_checkpoint()

        self.run_validation(True)

    @torch.no_grad()
    def run_validation(self, show_plot=False):
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
        logger.info(f'Epoch: {self.epoch:03} --> Validation Avg-loss: {mean_loss:9.6}')

        if show_plot:
            self._show_plot(pred_list, tgt_list)

    @torch.no_grad()
    def save_checkpoint(self):
        pass

    @torch.no_grad()
    def _show_plot(self, pred_slid_win, tgt_slid_win):

        def _sliding_win_to_seq(windows: list, component: int):
            windows = [torch.squeeze(w[:, :, component]) for w in windows]
            windows = torch.cat(windows, dim=0)
            ret = windows
            if windows.dim() == 2:
                ret = list(windows[0]) + list(windows[1:, 1])
            return [r.item() for r in ret]

        for idx, op_feat in enumerate(self._output_features):
            tgt_list = _sliding_win_to_seq(tgt_slid_win, idx)
            pred_list = _sliding_win_to_seq(pred_slid_win, idx)
            plt.plot(pred_list, label='Predicted')
            plt.plot(tgt_list, label='Actual')
            plt.title(self._name.upper() + '-' + op_feat.capitalize())
            plt.legend()
            plt.show()
            logger.info('Actual:   ' + str([f'{x:.3f} ' for x in tgt_list]))
            logger.info('Predicted:' + str([f'{x:1.3f} ' for x in pred_list]))

