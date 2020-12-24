import logging
import os
import pathlib
import urllib
from datetime import datetime, timedelta
from io import StringIO
from typing import List

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def collate_batch_fn(batch):
    xs = [i[0] for i in batch]
    ys = [i[1] for i in batch]
    return tuple([torch.stack(xs), torch.stack(ys)])


class CurrencyData(Dataset):

    ALL_FEATURES = ["high", "low", "open", "close", "average"]

    def __init__(self, data_dir: str, country: str,
                 num_samples: int, skip_past: int, frequency: str,
                 input_features: List[str], output_features: List[str],
                 input_seq_len: int, output_seq_len: int):
        super(CurrencyData, self).__init__()

        assert frequency.upper() == '1day'.upper(), 'Only 1 day sequences are supported in Currecy dataset currently'
        assert len(input_features) == len(output_features) == 1 and output_features[0] == 'average',\
            'Only \'average\' features are supported in Currency dataset currently'

        self.name = country.capitalize() + '-' + output_features[0]

        list_file = pathlib.Path(os.path.join(data_dir, 'country_list.csv'))
        if not list_file.exists():
            data = self._get_currency_data(data_dir, country, list_file)
        else:
            self._country_list = [r.strip().upper() for r in open(list_file).readlines()]
            country = country.upper()
            if country not in self._country_list:
                raise ValueError(f'"{country}" Not in list of : {self._country_list}')
            country_file = os.path.join(data_dir, country + '.csv')
            data = pd.read_csv(country_file).dropna()

        del data['Date']

        tot_days = num_samples + skip_past
        arr = data.to_numpy()[tot_days:]

        self._in_data = arr[-tot_days:] if skip_past == 0 else arr[-tot_days:-skip_past]

        self._input_scale = self._in_data.mean(0)
        self._in_data /= self._input_scale

        self._num_past = input_seq_len
        self._num_future = output_seq_len

    def scale(self, ip, op):
        if ip is not None:
            ip *= self._input_scale
        if op is not None:
            op *= self._input_scale
        return ip, op

    def __len__(self):
        return len(self._in_data) - (self._num_past + self._num_future)

    def __getitem__(self, index):
        end_1 = index + self._num_past
        end_2 = end_1 + self._num_future
        ip, op = self._in_data[index:end_1], self._in_data[index:end_2]
        return (torch.from_numpy(ip).float(), torch.from_numpy(op).float())

    def _get_currency_data(self, data_dir, country, list_file):

        pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
        country = country.upper()

        try:
            url = 'https://pkgstore.datahub.io/core/exchange-rates/daily_csv/data/03e15e28c7eea87026fee299a8859e97/daily_csv.csv'
            r = requests.get(url, verify=False)
        except urllib.error.HTTPError as err:
            logger.error(f'Could not read {url }for error: {err}')
            return None, "404"
        if r.status_code != 200 or len(r.text) == 0:
            logger.error(f'Could not read `{url}`')
            return None, "EmptyData"

        frame = pd.read_csv(StringIO(r.text))

        by_country = frame.groupby(frame.Country)

        ret_frame = None
        country_list = []
        for ctry in by_country.groups:
            ctry_frame = pd.DataFrame(frame[frame['Country'] == ctry])
            del ctry_frame['Country']
            ctry_frame.to_csv(os.path.join(data_dir, ctry.upper() + '.csv'), index=False)
            if country == ctry.upper():
                ret_frame = ctry_frame
            country_list.append(ctry)

        with open(list_file, 'w') as c_list:
            c_list.write('\n'.join(country_list))
        if ret_frame is None:
            raise ValueError(f'Country "{country}" not found')

        return ret_frame
