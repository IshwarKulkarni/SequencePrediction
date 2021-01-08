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

from dataset_loaders.base_dataset import TrainableDataset

logger = logging.getLogger(__name__)

class CurrencyData(TrainableDataset):

    ALL_FEATURES = ["high", "low", "open", "close", "average"]

    def __init__(self, data_dir: str, country: str, num_samples: int, skip_past: int, frequency: str,
                 input_features: List[str], output_features: List[str], input_seq_len: int,
                 output_seq_len: int, overlap: bool):

        assert frequency.upper() == '1day'.upper(), 'Only 1 day sequences are supported in Currecy dataset currently'
        assert len(input_features) == len(output_features) == 1 and output_features[0] == 'average',\
            'Only \'average\' feature is supported in Currency dataset currently'

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

        in_data = arr[-tot_days:] if skip_past == 0 else arr[-tot_days:-skip_past]

        super().__init__(in_data, in_data, input_seq_len, output_seq_len, overlap)

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
