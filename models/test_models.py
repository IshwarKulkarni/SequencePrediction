import pytest
import torch
import itertools
import encoder_decoder_recurrent as models

class_types = [models.SimpleRecurrentCell,
               models.EncoderDecoderRecurrentCell,
               models.EncoderDecoderRecurrent]

recurrence_types = ['rnn', 'gru', 'lstm']

sep_futures = [True, False]


@pytest.mark.parametrize('class_type, recurrence_type',
                         itertools.product(class_types, recurrence_types)
                         )
def test_instantiation(class_type, recurrence_type):

    print(f"Trying combo: {class_type} x {recurrence_type} ")

    batch_n, num_past, input_n_features = 4, 3, 1
    args = {
        "recurrent_type": recurrence_type,
        "rec_hidden_size": 160,
        "encoder_sizes": [16],
        "rec_ip_size": 16,
        "num_layers": 1,
        "decoder_sizes": [8],
        "future_enc_sizes": [8],
        "seperate_future_recurrent": True,
        "perturb_init": True,
        "input_n_features": input_n_features,
        "output_n_features": 1
    }

    model = class_type(**args)

    num_fut = 2
    in_tensor = torch.ones((batch_n, num_past, input_n_features), requires_grad=True)
    out = model( (in_tensor, num_fut) )
    loss = out[-1].sum()
    loss.backward()

    assert not torch.allclose(in_tensor.grad[-1], in_tensor.grad[-1] * 0)
    assert torch.allclose(in_tensor.grad[:-1], in_tensor.grad[:-1] * 0)
    