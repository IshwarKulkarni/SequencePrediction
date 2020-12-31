from typing import List
import logging

import torch
import torch.nn as nn

from misc.reflections import make_class_from_module

logger = logging.getLogger(__name__)


class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_n_features: int,
                 recurrent_ip_size: int, recurrent_hidden_size: int,
                 output_n_features: int, perturb_init: bool, **kwargs):
        super().__init__()

        self._encoder = self._make_seq_linear(input_n_features, recurrent_ip_size,
                                              kwargs.get('encoder_sizes', []))

        self.split_recurrent = kwargs.get('split_recurrent', False)

        self._past_rec = nn.LSTMCell(recurrent_ip_size, recurrent_hidden_size)
        self._future_rec = self._past_rec
        if self.split_recurrent:
            self._future_rec = nn.LSTMCell(recurrent_ip_size, recurrent_hidden_size)

        self._fut_encoder = self._make_seq_linear(recurrent_hidden_size, recurrent_ip_size,
                                                      kwargs.get('future_enc_sizes', []))
        self._encoder_activation = nn.Tanh()

        self._decoder = self._make_seq_linear(recurrent_hidden_size, output_n_features,
                                              kwargs.get('decoder_sizes', []))

        self._rnn_hidden_size = recurrent_hidden_size

        self._perturb_init = perturb_init
        self._perturb_std = 1.0/15

        self._log()

    def _log(self):
        k_v = [(k, sum(p.numel() for p in v.parameters())) for k, v in self._modules.items()]
        total, log_str = 0, ''
        for k, v in k_v:
            if v > 0 and not (self.split_recurrent or k == '_future_rec'):
                log_str += (', ' + k + ' ' + str(v))
                total += v

        logger.info(f'Model initialized with {total} parameters' + log_str)
        logger.debug(self)

    def _make_seq_linear(self, in_size, out_size, stack_sizes):

        stack_sizes = [in_size] + stack_sizes + [out_size]

        block = []
        for i in range(0, len(stack_sizes)-1):
            block += [nn.Linear(stack_sizes[i], stack_sizes[i+1])]

        return nn.Sequential(*block)

    def forward(self, x):
        input_seq, future_n = x
        batch_n, past_n, *_ = input_seq.shape

        if self.training and self._perturb_init:
            state = (torch.randn(batch_n, self._rnn_hidden_size) * self._perturb_std,
                     torch.randn(batch_n, self._rnn_hidden_size) * self._perturb_std)
        else:
            state = (torch.zeros(batch_n, self._rnn_hidden_size),
                     torch.zeros(batch_n, self._rnn_hidden_size))

        encoded = self._encoder(input_seq)
        rec_op = []
        for t in range(past_n):
            state = self._past_rec(encoded[:, t, :], state)
            rec_op.append(state[0])

        for _ in range(future_n):
            last_out = self._fut_encoder(state[0])
            state = self._future_rec(last_out, state)
            rec_op.append(state[0])

        output = self._decoder(torch.stack(rec_op, 1))
        return output
