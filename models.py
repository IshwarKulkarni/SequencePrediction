import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EncoderDecoderLSTM(nn.Module):
    def __init__(self, ip_n_feat: int, encoder_szs: List[int],
                 recurrent_ip_size: int, rnn_hidden_size: int, num_layers: int,
                 decoder_szs: List[int], op_n_feat: int):
        super(EncoderDecoderLSTM, self).__init__()
        #self._encoder = self._make_seq_linear(ip_n_feat, encoder_szs, recurrent_ip_size) if \
            #len(encoder_szs) > 0 else nn.Identity()

        self._recurrent = nn.LSTMCell(recurrent_ip_size, rnn_hidden_size)

        self._decoder = self._make_seq_linear(rnn_hidden_size, decoder_szs, op_n_feat) if \
            len(encoder_szs) > 0 else nn.Sequential(nn.Linear(rnn_hidden_size, op_n_feat))
        self._num_layers = num_layers
        self._rnn_hidden_size = rnn_hidden_size
        self.log()

    def log(self):
        def ct_wt(module):
            count = 0
            for p in module.parameters():
                count += p.numel()
            return count
        enc_ct, dec_ct = 0, ct_wt(self._decoder)
        rec_ct = ct_wt(self._recurrent)
        ct = enc_ct + dec_ct + rec_ct
        logger.info(f'Model initialized with {ct} weights: encoder {enc_ct}, '
                    f'decoder {dec_ct}, recurrent: {rec_ct}')
        logger.debug(self)

    def _make_seq_linear(self, input_sz, layers_sizes, op_sz, dropout=.25):
        block = [nn.Linear(input_sz, layers_sizes[0]), nn.ReLU()]
        for sz in layers_sizes:
            block += [nn.Linear(block[-2].out_features, sz), nn.ReLU()]
        last_in_block = block[-2]
        if dropout > 0.0:
            block.append(nn.Dropout(.25))
        block += [nn.Linear(last_in_block.out_features, op_sz), nn.ReLU()]
        return nn.Sequential(*block)

    def forward(self, x):
        input_seq, future_n = x
        batch_n, past_n, *_ = input_seq.shape
        state = (torch.zeros(batch_n, self._rnn_hidden_size, dtype=torch.float32),
                 torch.zeros(batch_n, self._rnn_hidden_size, dtype=torch.float32))

        rec_op = []
        for t in range(past_n):
            time_slice = input_seq[:, t, :]
            state = self._recurrent(time_slice, state)
            output = self._decoder(state[0])
            rec_op.append(output)

        for _ in range(future_n):
            state = self._recurrent(output, state)
            output = self._decoder(state[0])
            rec_op.append(output)

        output = torch.stack(rec_op, 1)
        return output
