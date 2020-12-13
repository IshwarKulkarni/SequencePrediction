import torch.nn as nn
import torch
import logging

logger = logging.getLogger(__name__)


class EncoderDecoderLSTM(nn.Module):
    def __init__(self,
                 input_features_sz=7, encoder_szs=[2, 4],
                 recurent_ip_sz=16, rnn_hidden_size=16, num_layers=1,
                 decoder_szs=[4, 2], output_features_sz=2,
                 batch_size=8):
        super().__init__()
        self._encoder = self._make_seq_linear(
            input_features_sz, encoder_szs, recurent_ip_sz)
        self._recurrent = nn.LSTM(
            recurent_ip_sz, rnn_hidden_size, num_layers=num_layers, batch_first=True)
        self._decoder = self._make_seq_linear(
            rnn_hidden_size, decoder_szs, output_features_sz)
        self.log()
        hidden = torch.randn(num_layers, batch_size, rnn_hidden_size)
        cell = torch.randn(num_layers, batch_size, rnn_hidden_size)
        torch.nn.init.xavier_uniform_(hidden)
        torch.nn.init.xavier_uniform_(cell)

        self._state = (hidden, cell)

    def log(self):
        def ct_wt(module):
            count = 0
            for p in module.parameters():
                count += p.numel()
            return count
        enc_ct, dec_ct, rec_ct = ct_wt(self._encoder), ct_wt(
            self._decoder), ct_wt(self._recurrent)
        ct = enc_ct + dec_ct + rec_ct
        logger.info(
            f'Model initialized with {ct} weights: encoder {enc_ct}, decoder {dec_ct}, recurrent: {rec_ct}')
        logger.info(self)

    def _make_seq_linear(self, input_sz, layers_sizes, output_features_sz, dropout=.25):
        block = [nn.Linear(input_sz, layers_sizes[0]), nn.ReLU()]
        for sz in layers_sizes:
            block += [nn.Linear(block[-2].out_features, sz), nn.ReLU()]
        last_in_block = block[-2]
        if dropout > 0.0:
            block.append(nn.Dropout(.25))
        block += [nn.Linear(last_in_block.out_features, output_features_sz), nn.ReLU()]
        return nn.Sequential(*block)

    def forward(self, input_seq):

        encoded = self._encoder(input_seq)
        lstm_out, self._state = self._recurrent(encoded, self._state)
        self._state[0].detach_()
        self._state[1].detach_()
        return self._decoder(lstm_out)


if __name__ == "__main__":
    model = EncoderDecoderLSTM()
    print(model)
