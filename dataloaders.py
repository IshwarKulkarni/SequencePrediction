import logging
import os
import pathlib
import urllib
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import pathlib
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

    def __init__(self, cl_args, data_dir: str, country: str,
                 num_days_to_fetch: int, num_days_to_skip: int,
                 input_seq_len: int, output_seq_len: int,
                 **kwargs):
        super(CurrencyData, self).__init__()
        list_file = pathlib.Path(os.path.join(data_dir, 'country_list.csv'))
        if not list_file.exists():
            frame = self._get_currency_data(data_dir, country, list_file)
        else:
            self._country_list = [r.strip().upper() for r in open(list_file).readlines()]
            country = country.upper()
            if country not in self._country_list:
                raise ValueError(f'"{country}" Not in list of : {self._country_list}')
            country_file = os.path.join(data_dir, country + '.csv')
            frame = pd.read_csv(country_file).dropna()

        #frame['Date'] = pd.to_datetime(frame['Date'])
        # frame = resample(frame)
        del frame['Date']
        tot_days = num_days_to_fetch + num_days_to_skip
        arr = frame.to_numpy()[tot_days:]
        self._in_data = arr[-tot_days:] if num_days_to_skip == 0 else\
            arr[-tot_days:-num_days_to_skip]

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


class IEXDataset(Dataset):

    ALL_FEATURES = ["high", "low", "open", "close",
                    "average", "volume", "numberOfTrades"]

    def __init__(self, cl_args, tickers: str, data_dir: str, shuffle_rows:bool,
                 num_days_to_fetch: int, num_days_to_skip: int,
                 input_features: list, output_features: list,
                 input_seq_len: int, output_seq_len: int, frequency: str):

        super(IEXDataset, self).__init__()
        if hasattr(cl_args, 'IEX_token'):
            self._IEX_TOKEN = cl_args.IEX_token
        elif 'IEX_Token' in os.environ:
            self._IEX_TOKEN = os.environ['IEX_Token']
        else:
            raise EnvironmentError('IEX_Token not found in args or envirnment varaibles')
        self._data_dir = data_dir
        self._date_readble_fmt = '%Y/%m/%d'
        self._date_IEX_fmt = '%Y%m%d'
        self._date_fn_fmt = '%Y_%m_%d'
        self._date_attempts = 10

        self._freq = frequency

        bad_features = [f for f in input_features if f not in IEXDataset.ALL_FEATURES]
        assert not bad_features, f"Some unrecognized input features requested: {bad_features}."
        bad_features = [f for f in output_features if f not in IEXDataset.ALL_FEATURES]
        assert not bad_features, f"Some unrecognized output features requested: {bad_features}."

        frames = []
        for ticker in tickers.split(','):
            frames.append(self._get_intra_day(ticker.strip().upper(), 
                                              frequency=frequency,
                                              num_days_to_skip=num_days_to_skip,
                                              num_days_to_fetch=num_days_to_fetch))
        data = pd.concat(frames).dropna()
        if shuffle_rows:
            data = data.sample(frac=1).reset_index(drop=True)

        self._in_data = data[input_features].to_numpy().astype(np.float32)
        self._out_data = data[output_features].to_numpy().astype(np.float32)

        self._input_scale = self._in_data.mean(0)
        self._output_scale = self._out_data.mean(0)

        self._in_data /= self._input_scale
        self._out_data /= self._output_scale

        self._num_past = input_seq_len
        self._num_future = output_seq_len

    def __len__(self):
        return self._in_data.shape[0] - (self._num_past + self._num_future)

    def __getitem__(self, index):
        end_1 = index + self._num_past
        end_2 = end_1 + self._num_future
        ip, op = self._in_data[index:end_1], self._out_data[index:end_2]
        assert not np.isnan(ip).any() and not np.isnan(op).any()
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
        ticker = ticker.upper()
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

    def _get_intra_day(self, ticker,
                       num_days_to_skip: int,  # Days in past, starting today, to skip
                       num_days_to_fetch: int,  # walk back until these many days of data is accrued
                       frequency: str):
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
                frames.append(self._resample(frame, frequency))

        ret_frame = pd.concat(frames[num_days_to_skip:])
        logger.info(f'DataGen fetched {len(ret_frame)} rows')

        return ret_frame
