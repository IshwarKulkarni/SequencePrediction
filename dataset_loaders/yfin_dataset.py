import logging
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import requests
from parse import parse

from dataset_loaders.base_dataset import TrainableDataset

logger = logging.getLogger(__name__)


class YFinDataset(TrainableDataset):
    """A dataset based on Yahoo! API, it's fast but min frequency is only 60Min or higher after 60 days.
       Also, this class does not cache any results, thus does a web query each time this is used."""
    FREQS = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']

    ALL_FEATURES = ["high", "low", "open", "close"]

    def __init__(self, ticker: str, num_days: int, frequency: str, skip_days: int,
                 input_features: List[str], output_features: List[str], pre_post_mkt: bool,
                 input_seq_len: int, output_seq_len: int, overlap: bool, **kwargs):

        self.name = ticker.capitalize()

        def assert_in(items: List[str], name: str, allowed: List[str]):
            for item in items:
                assert item.lower() in allowed, f'Invalid {name}: {item}, valid values: {",".join(allowed)}'

        frequency = frequency.lower().replace('min', 'm')
        input_features = [v.strip() for v in input_features]
        output_features = [v.strip() for v in output_features]
        assert_in([frequency], 'frequency', YFinDataset.FREQS)
        assert_in(input_features, 'input_features', YFinDataset.ALL_FEATURES)
        assert_in(output_features, 'output_features', YFinDataset.ALL_FEATURES)

        def decrement_to_workday(start: datetime, dec: int):
            while dec > 0:
                # Skip wekeends, christmas, new years, indep day etc.
                while (start.weekday() >= 5) or (start.month == 12 and start.date == 25) or\
                      (start.month == 1 and start.date == 1) or (start.month == 7 and start.date == 4):
                    start -= timedelta(days=1)
                start -= timedelta(days=1)
                dec -= 1
            return start

        assert skip_days >= 0, "Data fetching cannot begin from future"
        end = decrement_to_workday(datetime.now(), skip_days)
        start = decrement_to_workday(end, num_days)
        prepost = pre_post_mkt and not('volume' in input_features and 'volume' in output_features)
        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?symbol={ticker}' +\
              f'&period1={int(start.timestamp())}&period2={int(end.timestamp())}&interval={frequency}&' +\
              f'includePrePost={prepost}&lang=en-US&region=US&crumb=t5QZMhgytYZ'

        result = requests.get(url)
        assert result.status_code == 200, f"Query failed: {result.reason}"
        chart = result.json()['chart']
        assert chart['error'] is None,  f"Yahoo! Finance error: {chart['error']['description']}"

        data = pd.DataFrame.from_dict(chart['result'][0]['indicators']['quote'][0]).dropna()

        start_s, end_s = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
        logging.info(f'Fetched {(end - start).days} days of "{ticker.upper()}" from {start_s} '
                     f'till {end_s} resulting in {len(data)} rows')

        in_data = data[input_features].to_numpy().astype(np.float32)
        out_data = data[output_features].to_numpy().astype(np.float32)

        super().__init__(in_data, out_data, input_seq_len, output_seq_len, overlap)
