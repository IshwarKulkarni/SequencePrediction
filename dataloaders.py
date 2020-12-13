import json
import logging
import os
import pathlib
import urllib
from datetime import datetime, timedelta
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class IEXTickerGen():

    TOKEN = 'pk_75e07a76785a412da7b5010100409c44'
    ALL_FEATURES = ['high', 'low', 'open', 'close', 'average']

    def __init__(self, data_dir='./data'):
        self._data_dir = data_dir
        self._date_readble_fmt = '%Y/%m/%d'
        self._date_IEX_fmt = '%Y%m%d'
        self._date_fn_fmt = '%Y_%m_%d'
        self._date_attempts = 10

    def _get_from_IEX(self, date, ticker: str):
        """Fetch the data from IEX and clean up a bit"""
        date_str = date.strftime(self._date_IEX_fmt)
        readble_str = date.strftime(self._date_readble_fmt)

        try:
            url1 = f'https://cloud.iexapis.com/stable/stock/{ticker}/chart/date/'
            url2 = f'{date_str}?format=csv&token={IEXTickerGen.TOKEN}&chartIEXOnly=true'
            url = url1 + url2
            r = requests.get(url)
        except urllib.error.HTTPError as err:
            logger.error(
                f'Could not read `{ticker}` for {readble_str}, error: {err}')
            return None, "404"
        if r.status_code != 200 or len(r.text) == 0:
            logger.error(
                f'Could not read `{ticker}` for {readble_str}, Market closed?')
            return None, "EmptyData"

        frame = pd.read_csv(StringIO(r.text))
        frame['datetime'] = frame['date'] + '-' + frame['minute']
        del frame['date'], frame['minute'], frame['label'], frame['symbol'], frame['notional']
        return frame, "OK"

    def _get_day_values(self, date, ticker: str):
        """"Eiether read a file on disk or fetch from API(and write to disk)"""
        if date.weekday() >= 5:  # Weekend
            return None, "Weekend"
        ticker_dir = pathlib.Path(f'{self._data_dir}/{ticker}')
        file_path = os.path.join(
            ticker_dir, date.strftime(self._date_fn_fmt)+'.csv')

        if os.path.exists(file_path):
            frame = pd.read_csv(file_path)
        else:
            frame, reason = self._get_from_IEX(date, ticker)
            if frame is None:
                return frame, reason
            ticker_dir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(file_path, index=False)  # how about .to_parquet

        frame['datetime'] = pd.to_datetime(
            frame['datetime'], format='%Y-%m-%d-%H:%M')

        return frame, "OK"

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

    def get_data(self, ticker,
                 num_days_to_skip=0,  # Days in past, starting today, to skip
                 num_days_to_fetch=10,  # walk back until these many days of data is accrued
                 freq='15Min'):  # fequency
        assert num_days_to_skip >= 0, f'Fetching cannot end in future'
        date = datetime.now()
        frames = []
        attempts_left = (num_days_to_fetch + num_days_to_skip) * 4
        while len(frames) < (num_days_to_fetch + num_days_to_skip) and attempts_left >= 0:
            frame, _ = self._get_day_values(date, ticker)
            date -= timedelta(days=1)
            attempts_left -= 1
            if frame is None:
                continue
            # we're wasting some fetches here, but should be small.
            if len(frames) < num_days_to_skip:
                frames.append(None)
            else:
                frames.append(self._resample(frame, freq))

        ret_frame = pd.concat(frames[num_days_to_skip:])

        return ret_frame


class IEXDataset(Dataset):
    def __init__(self, ticker, num_days_to_fetch: int, num_days_to_skip: int,
                 input_features: list, output_features: list,
                 input_seq_sz: int, output_seq_size: int, freq='15Min'):

        super().__init__()
        self._datagen = IEXTickerGen()

        bad_features = [f for f in input_features if f not in IEXTickerGen.ALL_FEATURES]
        assert not bad_features, f"Some unrecognized input features requested: {bad_features}."
        bad_features = [f for f in output_features if f not in IEXTickerGen.ALL_FEATURES]
        assert not bad_features, f"Some unrecognized output features requested: {bad_features}."

        gen_data = self._datagen.get_data(ticker, num_days_to_skip=num_days_to_skip,
                                          num_days_to_fetch=num_days_to_fetch, freq=freq)
        self._in_data = gen_data[input_features].to_numpy()
        self._out_data = gen_data[output_features].to_numpy()
        self._num_past = input_seq_sz
        self._num_future = output_seq_size
        self._future_padding = np.zeros(
            (output_seq_size, self._in_data.shape[1]))

    def __len__(self):
        return self._in_data.shape[0] - (self._num_past + self._num_future)

    def __getitem__(self, index):
        ip = np.concatenate(
            (self._in_data[index:(index + self._num_past)], self._future_padding))
        ip = torch.from_numpy(ip/100)
        op = torch.from_numpy(self._out_data[index:(
            index + self._num_past + self._num_future)])
        return (ip.to(torch.float32), op.to(torch.float32)/100)

    def collate_batch(self, batch):
        xs = [i[0] for i in batch]
        ys = [i[1] for i in batch]
        t = tuple([torch.stack(xs), torch.stack(ys)])
        return t


if __name__ == '__main__':
    ticker = 'AAPL'
    dataloader = IEXTickerGen()
    data = dataloader.get_data(ticker, 4)
    close = data['raw_frame']['close']
    ax = close.plot(title=f'{ticker}')
    ax.set_xlabel('date')
    ax.set_ylabel('close price')
    ax.grid()
    plt.show()
