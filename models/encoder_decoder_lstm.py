from typing import List, Tuple
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def make_lin_seq(in_size: int, out_size: int, stack_sizes: List[int]):
    """Make a sequential module of linear transforms."""
    stack_sizes = [in_size] + stack_sizes + [out_size]

    block = []
    for i in range(0, len(stack_sizes)-1):
        block += [nn.Linear(stack_sizes[i], stack_sizes[i+1])]

    return nn.Sequential(*block)


def _create_state(perturb_init: bool, shape: Tuple[int], perturb_std: float = 1.0/15):
    if perturb_init:
        return (torch.randn(shape) * perturb_std, torch.randn(shape) * perturb_std)

    return (torch.zeros(shape), torch.zeros(shape))


def _log_module(moudel_obj):
    assert all([k[0] == '_' for k in moudel_obj.keys()]), 'Parameters need to start with "_"'
    k_v = [(k[1:], sum(p.numel() for p in v.parameters())) for k, v in moudel_obj.items()]
    log_str = f'Model initialized with {sum(v for _, v in k_v):,} parameters'
    log_str += ''.join([f', {k}: {v:,} params' for k, v in k_v])
    logger.info(log_str)


class SimpleLSTMCell(nn.Module):
    """Model with 1 LSTM Cell for encoding the past sequence
        separate_future_lstm : set to True if a separate cell for encoding future sequence is needed.
    """

    def __init__(self, input_n_features: int, rec_hidden_size: int,
                 output_n_features: int, perturb_init: bool, separate_future_lstm: bool = True):
        super().__init__()

        assert input_n_features == output_n_features == 1

        self._past_rec = nn.LSTMCell(1, rec_hidden_size)
        if separate_future_lstm:
            self._future_rec = nn.LSTMCell(1, rec_hidden_size)
        else:
            self._future_rec = self._past_rec
            logger.info('Sharing LSTM weights for past and future encoding')
        self._decoder = nn.Linear(rec_hidden_size, 1)

        self._perturb_init = perturb_init
        self._rnn_hidden_size = rec_hidden_size
        _log_module(self._modules)
        logger.debug(self)

    def forward(self, x):
        """Forward method override."""
        input_seq, future_n = x
        batch_n, past_n, *_ = input_seq.shape
        state = _create_state(self.training and self._perturb_init, (batch_n, self._rnn_hidden_size))

        rec_op = []
        for t in range(past_n):
            state = self._past_rec(input_seq[:, t, :], state)
            rec_op.append(state[0])

        for _ in range(future_n):
            state = self._future_rec(self._decoder(state[1]), state)
            rec_op.append(state[0])

        output = self._decoder(torch.stack(rec_op, 1))
        return output


class EncoderDecoderLSTMCell(nn.Module):
    """Model with 1 LSTM Cell for encoding the past sequence
        separate_future_lstm : set to True if a separate LSTM cell for future is needed."""

    def __init__(self, input_n_features: int, rec_ip_size: int, rec_hidden_size: int,
                 output_n_features: int, perturb_init: bool, **kwargs):
        super().__init__()

        self._encoder = make_lin_seq(input_n_features, rec_ip_size, kwargs.get('encoder_sizes', []))
        self._past_rec = nn.LSTMCell(rec_ip_size, rec_hidden_size)

        self._future_rec = self._past_rec
        if kwargs.get('separate_future_lstm', False):
            self._future_rec = nn.LSTMCell(input_n_features, rec_hidden_size)

        self._fut_encoder = make_lin_seq(rec_hidden_size, rec_ip_size, kwargs.get('future_enc_sizes', []))
        self._decoder = make_lin_seq(rec_hidden_size, output_n_features, kwargs.get('decoder_sizes', []))

        self._rnn_hidden_size = rec_hidden_size

        self._perturb_init = perturb_init

        if kwargs.get('log_self', True):
            _log_module(self._modules)
            logger.debug(self)

    def forward(self, x):
        """Forward method override."""
        input_seq, future_n = x
        batch_n, past_n, *_ = input_seq.shape

        state = _create_state(self.training and self._perturb_init, (batch_n, self._rnn_hidden_size))

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


class EncoderDecoderLSTM(EncoderDecoderLSTMCell):
    """Model with 1 LSTM module for encoding the past sequence (can have > 1  layers)
        and 1 LSTM Cell for encoding future is needed."""

    def __init__(self, input_n_features: int,
                 rec_ip_size: int, rec_hidden_size: int,
                 output_n_features: int, perturb_init: bool, **kwargs):
        kwargs.update({'log_self': False})  # skip logging in super
        super().__init__(input_n_features, rec_ip_size, rec_hidden_size,
                         output_n_features, perturb_init, **kwargs)

        self._num_layers = kwargs.get('num_layers', 1)
        self._past_rec = nn.LSTM(rec_ip_size, rec_hidden_size,
                                 batch_first=True, num_layers=self._num_layers)
        self._future_rec = nn.LSTMCell(rec_ip_size, rec_hidden_size)

        _log_module(self._modules)
        logger.debug(self)

    def forward(self, x):
        input_seq, future_n = x
        batch_n, *_ = input_seq.shape

        state_shape = (self._num_layers, batch_n, self._rnn_hidden_size)
        state = _create_state(self.training and self._perturb_init, state_shape)

        encoded = self._encoder(input_seq)
        output, state = self._past_rec(encoded, state)
        rec_op = [output]

        state = tuple(s[-1] for s in state)  # pick the state from last layer.

        for _ in range(future_n):
            last_out = self._fut_encoder(state[0])
            state = self._future_rec(last_out, state)
            rec_op.append(state[0].unsqueeze(1))  # Add back the time dim

        output = self._decoder(torch.cat(rec_op, 1))
        return output
