from typing import List
import logging
import os
import pathlib
import urllib
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import requests
import torch

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)


logger = logging.getLogger(__name__)


class IEXDataset:

    ALL_FEATURES = ["high", "low", "open", "close", "average", "volume", "numberOfTrades"]

    def __init__(self, data_dir: str, ticker: str,
                 num_samples: int, skip_past: int, frequency: str,
                 input_features: List[str], output_features: List[str],
                 input_seq_len: int, output_seq_len: int):
        super().__init__()

        if 'IEX_Token' in os.environ:
            self._IEX_TOKEN = os.environ['IEX_Token']
        else:
            raise EnvironmentError('IEX_Token not found in args or envirnment varaibles')

        self.name = ticker.capitalize()

        self._data_dir = data_dir
        self._date_readble_fmt = '%Y/%m/%d'
        self._date_IEX_fmt = '%Y%m%d'
        self._date_fn_fmt = '%Y_%m_%d'

        self.ticker = ticker

        self.ticker_dir = pathlib.Path(f'{self._data_dir}/{ticker.upper()}')
        self.ticker_dir.mkdir(parents=True, exist_ok=True)

        input_features = [v.strip().lower() for v in input_features]
        output_features = [v.strip().lower() for v in output_features]

        assert skip_past >= 0, "Data fetching cannot begin from future"
        bad_features = [f for f in input_features if f not in IEXDataset.ALL_FEATURES]
        assert not bad_features, f"Some unrecognized input features requested: {bad_features}."
        bad_features = [f for f in output_features if f not in IEXDataset.ALL_FEATURES]
        assert not bad_features, f"Some unrecognized output features requested: {bad_features}."

        data = self._get_intra_day(ticker, frequency=frequency, num_days_to_skip=skip_past, num_samples=num_samples)

        self.date_range = (min(data.index[0], data.index[-1]), max(data.index[0], data.index[-1]))

        logger.info(f'Datset fetched {len(data)} rows, spanning {self.date_range[0]} to {self.date_range[1]}')
        self._in_data = data[input_features].to_numpy().astype(np.float32)
        self._out_data = data[output_features].to_numpy().astype(np.float32)

        self._input_scale = self._in_data.mean(0)
        self._output_scale = self._out_data.mean(0)

        self._in_data /= self._input_scale
        self._out_data /= self._output_scale

        self._input_seq_len = input_seq_len
        self._output_seq_len = output_seq_len

    def __len__(self):
        return self._in_data.shape[0] - (self._input_seq_len + self._output_seq_len)

    def __getitem__(self, index):
        end_1 = index + self._input_seq_len
        end_2 = end_1 + self._output_seq_len
        ip, op = self._in_data[index:end_1], self._out_data[index:end_2]
        return (torch.from_numpy(ip).float(), torch.from_numpy(op).float())

    def scale(self, ip, op):
        if ip is not None:
            ip *= self._input_scale
        if op is not None:
            op *= self._output_scale
        return ip, op

    def _get_from_IEX(self, date, ticker: str):
        """Fetch the data from IEX and clean up a bit"""
        date_str = date.strftime(self._date_IEX_fmt)
        readble_str = date.strftime(self._date_readble_fmt)

        try:
            url1 = f'https://cloud.iexapis.com/stable/stock/{ticker}/chart/date/'
            url2 = f'{date_str}?format=csv&token={self._IEX_TOKEN}'
            url = url1 + url2
            r = requests.get(url, verify=False)
        except urllib.error.HTTPError as err:
            logger.error(f'Could not read `{ticker}` for {readble_str}, error: {err}')
            return None, "404"
        if r.status_code != 200 or len(r.text) == 0:
            logger.error(f'Could not read `{ticker}` for {readble_str}, Market closed?')
            return None, "EmptyData"

        frame = pd.read_csv(StringIO(r.text))
        frame['datetime'] = frame['date'] + '-' + frame['minute']
        del frame['date'], frame['minute'], frame['label'], frame['symbol'], frame['notional']
        return frame, "OK"

    def _get_day_values(self, date, ticker: str):
        if date.weekday() >= 5:  # Weekend
            return None, "Weekend"
        ticker = ticker.upper()
        file_path = os.path.join(self.ticker_dir, date.strftime(self._date_fn_fmt)+'.csv')

        if os.path.exists(file_path):
            frame = pd.read_csv(file_path)
        else:
            frame, reason = self._get_from_IEX(date, ticker)
            if frame is None:
                return frame, reason
            
            frame.to_csv(file_path, index=False)  # how about .to_parquet

        frame['datetime'] = pd.to_datetime(frame['datetime'], format='%Y-%m-%d-%H:%M')
        return frame, "ok"

    def _resample(self, frame, freq='15Min'):
        rframe = pd.DataFrame()
        frame = frame.set_index('datetime').resample(freq)
        rframe['high'] = frame['high'].max()
        rframe['low'] = frame['high'].min()
        rframe['open'] = frame['open'].first()
        rframe['close'] = frame['close'].last()
        rframe['average'] = frame['average'].mean()
        rframe['volume'] = frame['volume'].sum()
        rframe['numberOfTrades'] = frame['numberOfTrades'].sum()
        return pd.DataFrame(rframe)

    def _get_intra_day(self, ticker, frequency: str,
                       num_days_to_skip: int,  # Days in past, starting today, to skip
                       num_samples: int):  # walk back until these many sampled are accumulated
        assert num_days_to_skip >= 0, 'Fetching cannot end in future'
        date = datetime.now()
        frames = []
        attempts_left = (num_samples + num_days_to_skip) * 4
        num_samples_remain = num_samples
        while num_samples_remain >= 0 and attempts_left >= 0:
            frame, _ = self._get_day_values(date, ticker)
            date -= timedelta(days=1)
            attempts_left -= 1
            if frame is None or len(frame) == 0:
                continue
            # we're wasting some fetches here, but should be small.
            if len(frames) < num_days_to_skip:
                frames.append(None)
            else:
                frame = self._resample(frame, frequency).dropna()
                num_samples_remain -= len(frame)
                frames.append(frame)

        ret_frame = pd.concat(frames[num_days_to_skip:])
        return ret_frame
