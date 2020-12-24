import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_n_features: int, recurrent_ip_size: int,
                 rnn_hidden_size: int, output_n_features: int, **kwargs):
        super().__init__()

        self._encoder = nn.Linear(input_n_features, recurrent_ip_size)

        self._recurrent = nn.LSTMCell(recurrent_ip_size, rnn_hidden_size)

        self._decoder = nn.Linear(rnn_hidden_size, output_n_features)

        self._rnn_hidden_size = rnn_hidden_size
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

    def _make_seq_linear(self, input_sz, layers_sizes, output_sz, dropout):

        if len(layers_sizes) == 0:
            return

        block = [nn.Linear(input_sz, layers_sizes[0]), nn.ReLU()]

        for sz in layers_sizes:
            block += [nn.Linear(block[-2].out_features, sz), nn.ReLU()]
        last_in_block = block[-2]

        if dropout > 0.0:
            block.append(nn.Dropout(.25))

        block += [nn.Linear(last_in_block.out_features, output_sz), nn.ReLU()]
        return nn.Sequential(*block)

    def forward(self, x):
        input_seq, future_n = x
        batch_n, past_n, *_ = input_seq.shape

        if self.training:
            state = (torch.randn(batch_n, self._rnn_hidden_size)/15,
                     torch.randn(batch_n, self._rnn_hidden_size)/15)
        else:
            state = (torch.zeros(batch_n, self._rnn_hidden_size),
                     torch.zeros(batch_n, self._rnn_hidden_size))

        encoded = self._encoder(input_seq)
        rec_op = []
        for t in range(past_n):
            rec_op.append(self._recurrent(encoded[:, t, :], state)[0])

        for _ in range(future_n):
            rec_op.append(self._recurrent(encoded[:, -1, :], state)[0])

        output = self._decoder(torch.stack(rec_op, 1))
        return output
