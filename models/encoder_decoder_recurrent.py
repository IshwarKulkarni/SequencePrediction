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


def get_recurrent_type(recurrent_type: str):
    """A shortcut instead of make_class_from_module."""
    accepted_types = {
        'lstm': (nn.LSTM, nn.LSTMCell, 2),
        'rnn': (nn.RNN, nn.RNNCell, 1),
        'gru': (nn.GRU, nn.GRUCell, 1)
    }
    recurrent_type = recurrent_type.lower()
    assert recurrent_type in accepted_types, f"recurrent type '{recurrent_type}' not " +\
                                             f"among {list(accepted_types.keys())}"
    return accepted_types[recurrent_type]


class SimpleRecurrentCell(nn.Module):
    """Model with 1 LSTM Cell for encoding the past sequence
        seperate_future_recurrent : set to True if a separate cell for encoding future sequence is needed.
    """

    def __init__(self, recurrent_type: str, input_n_features: int, rec_hidden_size: int,
                 output_n_features: int, perturb_init: bool, seperate_future_recurrent: bool, **kwargs):
        super().__init__()

        if type(self) == __class__:
            assert input_n_features == output_n_features == 1

        _, recurrent_cell_type, self._items_in_state = get_recurrent_type(recurrent_type)

        self._past_rec = recurrent_cell_type(input_n_features, rec_hidden_size)
        if seperate_future_recurrent:
            self._future_rec = recurrent_cell_type(input_n_features, rec_hidden_size)
        else:
            self._future_rec = self._past_rec
            logger.info('Sharing recurrent weights for past and future encoding')
        self._decoder = nn.Linear(rec_hidden_size, output_n_features)

        self._rnn_hidden_size = rec_hidden_size
        self._perturb_init = perturb_init

        if type(self) == __class__:  # isinstance does not work here
            self._log_module(recurrent_type)

    def _hidden(self, state):
        return state[0] if isinstance(state, tuple) else state

    def _create_state(self, shape: Tuple[int], perturb_std: float = 1.0/50):
        perturb = self._perturb_init & self.training
        gen, mult = (torch.randn, perturb_std) if perturb else (torch.ones, 0.01)

        def generator():
            return (gen(shape) * mult).requires_grad_()

        if self._items_in_state == 2:
            return generator(), generator()
        if self._items_in_state == 1:
            return generator()
        raise ValueError('Only 1 or 2 items in state are returned')

    def _log_module(self, rec_type):
        assert all([k[0] == '_' for k in self._modules.keys()]), 'Parameters need to start with "_"'
        k_v = {k[1:]: sum(p.numel() for p in v.parameters()) for k, v in self._modules.items()}
        if self._future_rec == self._past_rec:
            k_v.pop('future_rec')
        log_str = f'Model {type(self).__name__} with recurrent type "{rec_type}" '
        log_str += f'initialized with {sum(v for v in k_v.values()):,} parameters '
        log_str += ''.join([f', {k}: {v:,} params' for k, v in k_v.items()])
        logger.info(log_str)
        logger.debug(self)

    def forward(self, x):
        """Forward method override."""
        input_seq, future_n = x
        batch_n, past_n, *_ = input_seq.shape
        state = self._create_state((batch_n, self._rnn_hidden_size))

        rec_op = []
        for t in range(past_n):
            state = self._past_rec(input_seq[:, t, :], state)
            rec_op.append(self._hidden(state))

        for _ in range(future_n):
            state = self._future_rec(self._decoder(self._hidden(state)), state)
            rec_op.append(self._hidden(state))

        output = self._decoder(torch.stack(rec_op, 1))
        return output


class EncoderDecoderRecurrentCell(SimpleRecurrentCell):
    """Model with 1 LSTM Cell for encoding the past sequence
        seperate_future_recurrent : set to True if a separate LSTM cell for future is needed."""

    def __init__(self, recurrent_type: str, input_n_features: int, rec_ip_size: int,
                 rec_hidden_size: int, output_n_features: int, perturb_init: bool, **kwargs):
        super().__init__(recurrent_type, rec_ip_size, rec_hidden_size, output_n_features,
                         perturb_init, **kwargs)

        self._encoder = make_lin_seq(input_n_features, rec_ip_size, kwargs.get('encoder_sizes', []))
        self._fut_encoder = make_lin_seq(rec_hidden_size, rec_ip_size, kwargs.get('future_enc_sizes', []))
        self._decoder = make_lin_seq(rec_hidden_size, output_n_features, kwargs.get('decoder_sizes', []))

        if type(self) == __class__:
            self._log_module(recurrent_type)

    def forward(self, x):
        """Forward method override."""
        input_seq, future_n = x
        batch_n, past_n, *_ = input_seq.shape

        state = self._create_state((batch_n, self._rnn_hidden_size))

        encoded = self._encoder(input_seq)
        rec_op = []
        for t in range(past_n):
            state = self._past_rec(encoded[:, t, :], state)
            rec_op.append(self._hidden(state))

        for _ in range(future_n):
            last_out = self._fut_encoder(self._hidden(state))
            state = self._future_rec(last_out, state)
            rec_op.append(self._hidden(state))

        output = self._decoder(torch.stack(rec_op, 1))
        return output


class EncoderDecoderRecurrent(EncoderDecoderRecurrentCell):
    """Model with 1 LSTM module for encoding the past sequence (can have > 1  layers)
        and 1 LSTM Cell for encoding future is needed."""

    def __init__(self, recurrent_type: str, input_n_features: int, rec_ip_size: int,
                 rec_hidden_size: int, output_n_features: int, perturb_init: bool, **kwargs):
        super().__init__(recurrent_type, input_n_features, rec_ip_size, rec_hidden_size,
                         output_n_features, perturb_init, **kwargs)

        recurrent_class_type, _, self._items_in_state = get_recurrent_type(recurrent_type)

        self._num_layers = kwargs.get('num_layers', 1)
        dropout = kwargs.get('dropout', 0.0)
        self._past_rec = recurrent_class_type(rec_ip_size, rec_hidden_size, dropout=dropout,
                                              batch_first=True, num_layers=self._num_layers)

        if type(self) == __class__:
            self._log_module(recurrent_type)

    def forward(self, x):
        input_seq, future_n = x
        batch_n, *_ = input_seq.shape

        state_shape = (self._num_layers, batch_n, self._rnn_hidden_size)
        state = self._create_state(state_shape)

        encoded = self._encoder(input_seq)
        output, state = self._past_rec(encoded, state)
        rec_op = [output]

        # Pick the state from last layer:
        state = tuple(s[-1] for s in state) if isinstance(state, tuple) else state[-1]

        for _ in range(future_n):
            last_out = self._fut_encoder(self._hidden(state))
            state = self._future_rec(last_out, state)
            rec_op.append((self._hidden(state)).unsqueeze(1))  # Add back the time dim

        output = self._decoder(torch.cat(rec_op, 1))
        return output
