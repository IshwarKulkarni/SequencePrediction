from  typing import List
import logging

import torch
import torch.nn as nn

from misc.reflections import make_class_from_module

logger = logging.getLogger(__name__)


class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_n_features: int,
                 recurrent_ip_size: int, recurrent_hidden_size: int,
                 output_n_features: int, perturb_init:bool, **kwargs):
        super().__init__()

        self._encoder = self._make_seq_linear(input_n_features, kwargs.get('encoder_sizes', []),
                                              recurrent_ip_size)

        self._recurrent = nn.LSTMCell(recurrent_ip_size, recurrent_hidden_size)

        self._decoder = self._make_seq_linear(recurrent_hidden_size, kwargs.get('decoder_sizes', []),
                                              output_n_features)

        self._rnn_hidden_size = recurrent_hidden_size

        self._perturb_init = perturb_init

        self.log()

    def log(self):
        def ct_wt(module):
            return sum(p.numel() for p in module.parameters())
        enc_ct, dec_ct = ct_wt(self._encoder), ct_wt(self._decoder)
        rec_ct = ct_wt(self._recurrent)
        ct = enc_ct + dec_ct + rec_ct
        logger.info(f'Model initialized with {ct} weights: encoder {enc_ct}, '
                    f'decoder {dec_ct}, recurrent: {rec_ct}')
        logger.debug(self)

    def _make_seq_linear(self, in_size, stack_sizes, out_size):

        stack_sizes = [in_size] + stack_sizes + [out_size]

        block = []
        for i in range(0, len(stack_sizes)-1):
            block += [nn.Linear(stack_sizes[i], stack_sizes[i+1])]

        return nn.Sequential(*block)

    def forward(self, x):
        input_seq, future_n = x
        batch_n, past_n, *_ = input_seq.shape

        if self.training and self._perturb_init:
            state = (torch.randn(batch_n, self._rnn_hidden_size)/15,
                     torch.randn(batch_n, self._rnn_hidden_size)/15)
        else:
            state = (torch.zeros(batch_n, self._rnn_hidden_size),
                     torch.zeros(batch_n, self._rnn_hidden_size))

        encoded = self._encoder(input_seq)
        rec_op = []
        for t in range(past_n):
            state = self._recurrent(encoded[:, t, :], state)
            rec_op.append(state[0])

        for _ in range(future_n):
            state = self._recurrent(encoded[:, -1, :], state)
            rec_op.append(state[0])

        output = self._decoder(torch.stack(rec_op, 1))
        return output
